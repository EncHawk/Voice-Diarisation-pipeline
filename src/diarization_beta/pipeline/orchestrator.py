"""Pipeline orchestrator — offline batch processing (spec §24)."""
from __future__ import annotations

import json
import pathlib
import time
import uuid

import numpy as np


def _ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)


def process_file(
    audio_path: str | pathlib.Path,
    config: dict | None = None,
    output_dir: str | pathlib.Path | None = None,
    db_path: str | pathlib.Path | None = None,
    recording_id: str | None = None,
    skip_transcription: bool = False,
    skip_llm: bool = False,
) -> dict:
    """Run full pipeline on a single file. Returns result dict with paths."""
    from ..audio.loader import load_audio
    from ..audio.preprocess import normalize
    from ..config import load_config
    from ..diarization.clustering import cluster_embeddings
    from ..diarization.embeddings import EmbeddingExtractor
    from ..diarization.speaker_profile import build_profiles
    from ..entities.name_extractor import (
        extract_entities,
        heuristic_evidence,
        is_model_name_candidate,
        name_usage_supported,
    )
    from ..identity.confidence import should_display_name
    from ..identity.resolver import resolve_identities
    from ..reasoning.context_builder import (
        build_context,
        build_llm_prompt,
        build_name_listing_prompt,
    )
    from ..reasoning.evidence_extractor import evidence_to_resolver_format, merge_evidence
    from ..vad.detector import VADDetector

    if config is None:
        config = load_config()

    audio_path = pathlib.Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    # Resolve output locations
    cfg_out = config.get("pipeline", {}).get("output_dir", "outputs") if isinstance(config.get("pipeline"), dict) else "outputs"
    # config is yaml dict; load_config returns dict
    pipeline_cfg = config.get("pipeline", {}) if isinstance(config.get("pipeline"), dict) else {}
    default_out = pipeline_cfg.get("output_dir", "outputs") if isinstance(pipeline_cfg, dict) else "outputs"
    out_dir = pathlib.Path(output_dir) if output_dir else pathlib.Path(default_out) / audio_path.stem
    _ensure_dir(out_dir)

    storage_path = pathlib.Path(db_path) if db_path else pathlib.Path(config.get("storage", {}).get("db_path", "diarization.db") if isinstance(config.get("storage"), dict) else "diarization.db")

    t0 = time.time()

    # 1. Load audio
    print(f"[pipeline] loading {audio_path}")
    audio, sr = load_audio(audio_path)
    duration = len(audio) / sr
    print(f"[pipeline] audio: {duration:.1f}s @ {sr}Hz, {len(audio)} samples")
    audio = normalize(audio)

    # 2. VAD
    vad_cfg = config.get("vad", {}) if isinstance(config.get("vad"), dict) else {}
    vad = VADDetector(
        threshold=float(vad_cfg.get("threshold", 0.5)),
        min_speech_duration_ms=int(vad_cfg.get("min_speech_duration_ms", 250)),
        min_silence_duration_ms=int(vad_cfg.get("min_silence_duration_ms", 100)),
        speech_pad_ms=int(vad_cfg.get("speech_pad_ms", 30)),
        window_size_samples=int(vad_cfg.get("window_size_samples", 512)),
        sample_rate=sr,
        model=str(vad_cfg.get("model", "silero_vad")),
    )
    vad_segments = vad.detect(audio)
    print(f"[pipeline] VAD: {len(vad_segments)} speech segments")
    if not vad_segments:
        # fallback: treat whole file as speech if VAD finds nothing but audio is non-silent
        if np.max(np.abs(audio)) > 0.01:
            from ..vad.detector import VADSegment

            vad_segments = [VADSegment(0.0, duration, 0.5)]
            print("[pipeline] VAD found nothing, using whole file as fallback")

    # Merge close VAD segments (gap < 0.30s) to avoid over-fragmentation from TTS intra-utterance pauses
    # Inter-turn silence is ~0.5s in synthetic data, so 0.30 preserves turn boundaries while merging word-level gaps
    merged_vad = []
    for s in sorted(vad_segments, key=lambda x: x.start):
        if not merged_vad:
            merged_vad.append(s)
        else:
            prev = merged_vad[-1]
            gap = s.start - prev.end
            if gap < 0.30:
                prev.end = max(prev.end, s.end)
                prev.speech_probability = max(prev.speech_probability, s.speech_probability)
            else:
                merged_vad.append(s)
    if len(merged_vad) != len(vad_segments):
        print(f"[pipeline] VAD merged {len(vad_segments)} -> {len(merged_vad)} (gap<0.30s)")
        vad_segments = merged_vad

    # Persist VAD visualization
    vad_dicts = [s.to_dict() if hasattr(s, "to_dict") else {"start": s["start"], "end": s["end"]} for s in vad_segments]

    # 3. Embeddings
    emb_cfg = config.get("embeddings", {}) if isinstance(config.get("embeddings"), dict) else {}
    min_dur = float(emb_cfg.get("min_segment_duration", 1.0))
    space = str(emb_cfg.get("model", "auto"))
    if space == "auto":
        from ..diarization.embeddings import select_space

        space = select_space(audio, sr, vad_segments, clean_snr_threshold=float(emb_cfg.get("clean_snr_threshold", 150.0)))
    extractor = EmbeddingExtractor(
        model_path=emb_cfg.get("model_path", "models/ecapa.onnx"),
        dim=int(emb_cfg.get("dimension", 192)),
        space="spectral" if space == "spectral_fallback" or space == "spectral" else "ecapa",
    )
    records = extractor.embed_segments(audio, sr, vad_segments, min_duration=min_dur)
    print(f"[pipeline] embeddings: {len(records)} records (space={space})")

    if not records:
        # No embeddings (all segments too short etc.) — create single speaker fallback
        from ..diarization.embeddings import EmbeddingRecord

        # create one record covering whole file
        emb = extractor.embed_segment(audio, sr)
        records = [EmbeddingRecord(segment_id="seg_0000", start=0.0, end=duration, embedding=emb)]

    embeddings = np.stack([r.embedding for r in records], axis=0)

    # 4. Clustering
    clust_cfg = config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {}
    labels = cluster_embeddings(
        embeddings,
        distance_threshold=float(clust_cfg.get("distance_threshold", 0.35)),
        linkage=clust_cfg.get("linkage", "average"),
        metric=clust_cfg.get("metric", "cosine"),
    )
    print(f"[pipeline] clustering: {len(set(labels)) if len(labels) else 0} speakers -> {labels.tolist()[:20]}")

    # 5. Speaker profiles
    profiles = build_profiles(records, labels, embeddings)
    print(f"[pipeline] profiles: {list(profiles.keys())}")

    # 5b. Beta: voice-gender estimation from pitch (used for generic labels)
    gender_cfg = config.get("gender", {}) if isinstance(config.get("gender"), dict) else {}
    speaker_spans: dict[str, list[tuple[float, float]]] = {}
    for rec in records:
        speaker_spans.setdefault(rec.speaker_id, []).append((rec.start, rec.end))
    from ..diarization.gender import classify_speakers

    gender_map = classify_speakers(audio, sr, speaker_spans, gender_cfg)
    print(
        "[pipeline] gender: "
        + ", ".join(f"{sid}={info['gender']}({info['median_f0_hz']}Hz)" for sid, info in gender_map.items())
    )

    # Map VAD segments to speaker labels via records
    # records already linked to speaker_id via build_profiles
    diarized_segments = []
    for rec in records:
        diarized_segments.append({"segment_id": rec.segment_id, "speaker_id": rec.speaker_id, "start": rec.start, "end": rec.end})

    # Persist diarized segments for alignment
    # Also need to sort
    diarized_segments.sort(key=lambda x: x["start"])

    # 6. Transcription (whisper.cpp) — need a wav file for whisper-cli
    whisper_segments: list[dict] = []
    if skip_transcription:
        print("[pipeline] skipping transcription (flag)")
        # Create pseudo-transcript from silence: empty
        pass
    else:
        import tempfile

        # Write mono 16k wav temp for whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            from ..audio.loader import write_wav

            write_wav(tmp_path, audio, sr)
            trans_cfg = config.get("transcription", {}) if isinstance(config.get("transcription"), dict) else {}
            from ..transcription.whisper import transcribe

            whisper_segments = transcribe(
                tmp_path,
                model_path=trans_cfg.get("model_path", "models/ggml-base.en.bin"),
                whisper_cli=trans_cfg.get("whisper_cli", "whisper-cli"),
                language=trans_cfg.get("language", "en"),
                threads=int(trans_cfg.get("threads", 4)),
            )
            print(f"[pipeline] whisper: {len(whisper_segments)} segments")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    # If no whisper output but we have diarized segments, create placeholder aligned segments from diarized
    # (so pipeline still produces output for tests without model)
    aligned = []
    if whisper_segments:
        align_cfg = config.get("alignment", {}) if isinstance(config.get("alignment"), dict) else {}
        from ..alignment.timestamp_alignment import align as do_align

        aligned_objs = do_align(whisper_segments, diarized_segments, iou_threshold=float(align_cfg.get("iou_threshold", 0.1)))
        for a in aligned_objs:
            aligned.append({"segment_id": a.segment_id, "speaker_id": a.speaker_id, "start": a.start, "end": a.end, "text": a.text})
    else:
        # Fallback: use diarized segments with empty text; or if we can do a dummy transcription
        print("[pipeline] no whisper output — using diarized segments as transcript placeholders")
        for d in diarized_segments:
            aligned.append({"segment_id": d["segment_id"], "speaker_id": d["speaker_id"], "start": d["start"], "end": d["end"], "text": ""})

    # 7. Candidate listing and identity evidence
    # The generic extractor is only a fallback for --skip-llm or unavailable models.
    fallback_candidates = extract_entities(aligned)["entities"]
    fallback_names = [entity["name"] for entity in fallback_candidates]
    heuristic_evs = heuristic_evidence(aligned, fallback_names)
    supported_fallback_names = {
        event.get("name") or event.get("target_name") or event.get("candidate_name")
        for event in heuristic_evs
    }
    # Only retain fallback tokens that participate in actual evidence. This
    # prevents title-cased words such as "Nice" from becoming candidates.
    entities = [
        entity for entity in fallback_candidates if entity["name"] in supported_fallback_names
    ]

    # 8. LLM reasoning: first list names, then select only from that list
    llm_output = {"entities": [], "identity_hypotheses": []}
    reasoning_cfg = config.get("reasoning", {}) if isinstance(config.get("reasoning"), dict) else {}
    should_call_llm = bool(any(str(segment.get("text", "")).strip() for segment in aligned)) and not skip_llm
    if should_call_llm:
        # Profiles dict -> plain dict for context.
        prof_dict = {}
        for sid, p in profiles.items():
            if hasattr(p, "to_dict"):
                prof_dict[sid] = p.to_dict()
            else:
                prof_dict[sid] = p

        from ..reasoning.local_llm import LocalLLM

        llm = LocalLLM(
            model_path=reasoning_cfg.get("llm_model_path", "models/LFM2-700M-Q5_K_M.gguf"),
            llama_cli=reasoning_cfg.get("llama_cli", "llama-cli"),
            ctx_size=int(reasoning_cfg.get("ctx_size", 4096)),
            temp=float(reasoning_cfg.get("temp", 0.2)),
            n_predict=int(reasoning_cfg.get("n_predict", 512)),
            threads=int(reasoning_cfg.get("threads", 4)),
        )

        listed_names = []
        # Pass 1: the model decides which transcript words are actual names.
        # Process windows so a long recording cannot crowd its names out of context.
        listing_window = max(1, int(reasoning_cfg.get("name_listing_window_turns", 48)))
        for start in range(0, len(aligned), listing_window):
            name_listing = llm.generate(
                build_name_listing_prompt(
                    aligned[start : start + listing_window],
                    candidate_hints=fallback_names,
                ),
                mode="names",
            )
            for name in name_listing.get("names", []):
                if (
                    isinstance(name, str)
                    and is_model_name_candidate(name)
                    and name_usage_supported(name, aligned)
                    and name not in listed_names
                ):
                    listed_names.append(name)
        if listed_names:
            # Keep the LLM list authoritative, but recover a name it missed
            # when the generic fallback has independent turn evidence for it.
            for entity in fallback_candidates:
                name = entity["name"]
                if name in supported_fallback_names and name not in listed_names:
                    listed_names.append(name)
            entities = [{"name": name, "type": "person"} for name in listed_names]
        allowed_names = [entity["name"] for entity in entities]
        print(f"[pipeline] LFM name list: {listed_names}")
        heuristic_evs = heuristic_evidence(aligned, allowed_names)
        print(f"[pipeline] candidate names: {allowed_names}")
        print(f"[pipeline] heuristic evidence: {len(heuristic_evs)} items")

        # Pass 2: identity mapping is closed-world and can only select a listed name.
        ctx = build_context(prof_dict, aligned, entities)
        prompt = build_llm_prompt(ctx, heuristic_evs, aligned, allowed_names=allowed_names)
        selection = llm.generate(prompt, allowed_names=allowed_names)
        llm_output = {"entities": entities, "identity_hypotheses": selection.get("identity_hypotheses", [])}
        print(f"[pipeline] LLM hypotheses: {llm_output.get('identity_hypotheses', [])}")
    else:
        print(f"[pipeline] heuristic evidence: {len(heuristic_evs)} items, entities: {entities}")
        if skip_llm:
            print("[pipeline] LLM skipped (flag)")
        else:
            print("[pipeline] no transcript text for LLM")

    # 9. Merge evidence + resolve identities
    merged_evs = merge_evidence(heuristic_evs, llm_output, allowed_names=[entity["name"] for entity in entities])
    grouped = evidence_to_resolver_format(merged_evs)
    # also need to handle storage of previous confirmations
    identity_cfg = config.get("identity", {}) if isinstance(config.get("identity"), dict) else {}
    weights = identity_cfg.get("weights", {}) if isinstance(identity_cfg.get("weights"), dict) else {}
    thresholds = identity_cfg.get("thresholds", {}) if isinstance(identity_cfg.get("thresholds"), dict) else {}

    # Load previous confirmations for this recording if exists (do we have recording_id reuse?)
    # For batch run, previous is empty; user corrections persisted via storage
    # If skip transcription/LLM, still resolve on heuristic only

    rid = recording_id or audio_path.stem + "_" + uuid.uuid4().hex[:8]
    # Try to load confirmations from db
    storage = None
    try:
        from ..storage.sqlite import Storage

        storage = Storage(storage_path)
        confirmations = storage.get_confirmations(rid)
    except Exception:
        confirmations = {}

    hypotheses = resolve_identities(grouped, list(profiles.keys()), weights=weights, thresholds=thresholds, user_confirmations=confirmations)
    print(f"[pipeline] hypotheses: {hypotheses}")

    # 10. Apply identities to profiles + aligned segments (confidence gate)
    final_profiles = {}
    for sid, prof in profiles.items():
        hyp = hypotheses.get(sid, {"candidate_name": None, "confidence": 0.0, "status": "unknown"})
        cand = hyp.get("candidate_name")
        conf = float(hyp.get("confidence", 0))
        status = hyp.get("status", "unknown")
        if hasattr(prof, "to_dict"):
            d = prof.to_dict()
        else:
            d = dict(prof) if isinstance(prof, dict) else {"speaker_id": sid, "display_name": sid, "status": "unknown", "confidence": 0, "sample_count": 0, "segments": []}
        d["candidate_name"] = cand
        d["confidence"] = conf
        d["status"] = status
        # display_name: only if identified/confirmed
        if should_display_name(status) and cand:
            d["display_name"] = cand
            d["name"] = cand  # alias for output spec
        else:
            # keep Speaker N
            d["name"] = None
        d["evidence"] = hyp.get("evidence", [])
        d["gender"] = (gender_map.get(sid) or {}).get("gender")
        d["median_f0_hz"] = (gender_map.get(sid) or {}).get("median_f0_hz")
        final_profiles[sid] = d

    # Attach speaker_name to aligned segments (retroactive relabel per spec §17)
    from ..diarization.gender import assign_generic_labels

    first_start = {}
    for seg in sorted(aligned, key=lambda x: x["start"]):
        first_start.setdefault(seg["speaker_id"], seg["start"])
    generic_labels = assign_generic_labels(list(profiles.keys()), gender_map, first_start)
    final_segments = []
    for seg in aligned:
        sid = seg["speaker_id"]
        prof = final_profiles.get(sid, {})
        status = prof.get("status", "unknown")
        cand = prof.get("candidate_name")
        display = cand if should_display_name(status) and cand else generic_labels.get(sid, sid)
        if display.startswith("speaker_"):
            display = display.replace("speaker_", "Speaker ")
        final_segments.append(
            {
                "segment_id": seg["segment_id"],
                "speaker_id": sid,
                "speaker_name": display,
                "display_name": display,
                "status": status,
                "confidence": prof.get("confidence", 0),
                "gender": (gender_map.get(sid) or {}).get("gender"),
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
        )

    # Also need to sort final segments by start
    final_segments.sort(key=lambda x: x["start"])

    # 11. Persist
    if storage:
        try:
            storage.save_recording(rid, str(audio_path), duration, len(profiles))
            storage.save_segments(rid, final_segments)
            # convert final_profiles with centroid
            # need original profiles objects for centroids
            persist_profiles = {}
            for sid, fp in final_profiles.items():
                orig = profiles.get(sid)
                centroid = None
                if orig and hasattr(orig, "centroid_embedding"):
                    centroid = orig.centroid_embedding
                elif isinstance(orig, dict):
                    centroid = orig.get("centroid_embedding")
                # merge fp + centroid
                persist_profiles[sid] = {**fp, "centroid_embedding": centroid, "sample_count": fp.get("sample_count", len(fp.get("segments", [])))}
            storage.save_profiles(rid, persist_profiles)
            storage.save_embeddings(rid, records)
            storage.save_entities(rid, entities)
            storage.save_hypotheses(rid, hypotheses)
        except Exception as e:
            print(f"[pipeline] storage persist failed: {e}")
        finally:
            pass  # keep open? we close after
            # don't close yet if caller wants to use

    proc_time = time.time() - t0
    rtf = proc_time / duration if duration > 0 else 0
    print(f"[pipeline] done in {proc_time:.1f}s RTF={rtf:.2f}")

    # 12. Write outputs
    # JSON
    participants = []
    for sid, prof in final_profiles.items():
        participants.append(
            {
                "speaker_id": sid,
                "name": prof.get("candidate_name") if should_display_name(prof.get("status", "unknown")) else None,
                "display_name": prof.get("display_name", sid),
                "gender": prof.get("gender"),
                "median_f0_hz": prof.get("median_f0_hz"),
                "confidence": prof.get("confidence", 0),
                "status": prof.get("status", "unknown"),
            }
        )
    # For spec compliance, participants name is null if not identified
    # but also keep display_name as Speaker N

    output = {
        "recording_id": rid,
        "recording_path": str(audio_path),
        "duration": duration,
        "processing_time": proc_time,
        "rtf": rtf,
        "participants": participants,
        "segments": final_segments,
        "entities": entities,
        "evidence": merged_evs,
        "hypotheses": hypotheses,
        "vad_segments": vad_dicts,
        "diarized_segments": diarized_segments,
    }

    json_path = out_dir / "transcript.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Human-readable transcript (spec §18)
    txt_path = out_dir / "transcript.txt"
    with open(txt_path, "w") as f:
        for seg in final_segments:
            if not seg["text"]:
                continue
            # [HH:MM:SS] Name:
            s = seg["start"]
            hh = int(s // 3600)
            mm = int((s % 3600) // 60)
            ss = int(s % 60)
            stamp = f"[{hh:02d}:{mm:02d}:{ss:02d}]" if hh else f"[{mm:02d}:{ss:02d}]"
            # alternative per spec: [00:00:12]
            total = int(s)
            stamp2 = f"[{total//3600:02d}:{(total%3600)//60:02d}:{total%60:02d}]"
            f.write(f"{stamp2} {seg['speaker_name']}:\n{seg['text']}\n\n")

    # Also emit vad + diarization json for debugging
    debug_path = out_dir / "diarization.json"
    with open(debug_path, "w") as f:
        json.dump(
            {
                "vad": vad_dicts,
                "embeddings": [{"segment_id": r.segment_id, "speaker_id": r.speaker_id, "start": r.start, "end": r.end} for r in records],
                "profiles": {k: {kk: (v[kk].tolist() if kk == "centroid_embedding" and hasattr(v[kk], "tolist") else v[kk]) for kk in v if kk != "centroid_embedding"} if isinstance(v, dict) else str(v) for k, v in final_profiles.items()},
                "whisper": whisper_segments,
            },
            f,
            indent=2,
        )

    if storage:
        storage.close()

    return {
        "recording_id": rid,
        "duration": duration,
        "processing_time": proc_time,
        "rtf": rtf,
        "segments": final_segments,
        "participants": participants,
        "profiles": final_profiles,
        "output_dir": str(out_dir),
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "db_path": str(storage_path),
    }
