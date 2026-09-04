# Diarization Pipeline: VAD and harnessing transcripts to identify entities

## The Journey of a Recording

A recording passes through a line of stages. Each stage prepares something the next stage needs.

### 1. Audio Preparation

The recording is opened, converted to a single-channel stream at a fixed sampling rate, and levelled so that quiet and loud passages become comparable. This matters because every later stage behaves differently on uneven volume.

### 2. Voice Activity Detection (VAD)

Before anything clever happens, the app needs to know where people are actually speaking and where there is only silence, noise, or music. A small neural model called **Silero VAD** looks at short slices of sound and answers one question for each slice: "is someone talking here, yes or no?" The slices marked as speech are kept; the rest are thrown away. Very short gaps between slices are stitched together so one spoken sentence does not get chopped into artificial pieces.

The point of this stage is efficiency and cleanliness. Everything downstream only ever sees genuine speech.

### 3. Speaker Fingerprints (Embeddings)

Knowing that speech exists is not the same as knowing who is speaking. The app uses a small neural network — an **ECAPA-TDNN speaker encoder**, run locally through the ONNX runtime — that listens to a stretch of speech and turns the character of the voice into a short list of numbers, a "fingerprint". Voices that sound alike get fingerprints that sit close together; different voices sit far apart. This is the same idea as face recognition, but for voices.

Each speech slice gets its own fingerprint. Short, noisy slices are skipped because their fingerprints would be unreliable.

### 4. Grouping the Voices (Clustering)

The app then gathers all the fingerprints into groups. Each group is one physical person, a "speaker". From here on, the transcript is labelled with these anonymous speaker groups, numbered in order of appearance. A profile is kept for each group: a voice sample average, how much that person spoke, and when.

### 5. Classifying men and women (beta — it will not always be right)

While groups are being formed, the app also measures the pitch of each speaker's voice — the fundamental tone. Male voices sit lower, female voices sit higher, and a band in between is left honestly unclassified. Any speaker group that never gets a confident name is labelled "Man 1", "Woman 1", and so on in the order they first spoke, instead of the blander "Speaker 4". This is deliberately marked as beta: it is a statistical guide, not a fact, and it never overrides a real name.

### 6. Transcription (STT)

Now the words themselves. The app runs **Whisper**, converted to a fast local form, on the prepared audio. Whisper writes out the text with time marks. It is an English-first small model, chosen because it is accurate enough for meetings while still being quick on an ordinary processor.

### 7. Stitching Times Together (Alignment)

Whisper knows when words were said; the clustering knows who owned which stretch of sound. The alignment stage joins the two: every piece of text is married to the speaker group whose sound it overlaps most. After this step the transcript reads properly: a speaker, a time, and their words.

### 8. Finding Names

With a raw attributed transcript, the app hunts for names. A quick mechanical pass collects words that look like names. Then the local language model reads the transcript and lists every person's name it can find. Nothing from that list is trusted blindly: each proposed name must also be genuinely used like a name in the conversation — spoken to as an address, introduced, or claimed as "my name is". Sentence-starting ordinary words, place names, and stray punctuation artefacts are rejected by these checks, not by a list of recordings.

### 9. Deciding Who Is Who (Identity Reasoning)

The language model then does the reasoning pass: given the speakers, the transcript, and the collected clues, it proposes which name belongs to which voice. It is only allowed to choose from the verified list, so it cannot invent anyone. Alongside the model, simple structural rules contribute evidence: who was addressed, who answered after an address, who introduced themselves. All the evidence is weighed together, and a name is only displayed when the total confidence clears the bar. Otherwise the speaker keeps their generic label, because a wrong name is worse than an honest unknown.

### 10. Outputs

The finished work is written out as a human-readable transcript file and a full machine-readable record, plus a debug file. Everything is also stored in a small local database so recordings can be revisited, and so you can correct a speaker by hand ("this is actually Dan") and have that correction persist and win over any future guesses.

---

## The Models, and What Each One Is For

| Model | Job | Why this one |
| --- | --- | --- |
| Silero VAD | Temporal context for when someonoe speaks | Tiny, fast, accurate at the edges of speech; costs almost nothing to run |
| ECAPA-TDNN (ONNX) | Turn voices into comparable fingerprints | Purpose-built for speakers; small enough for a processor without a graphics card |
| Whisper (base English, local fast build) | Write the words | Good accuracy-to-speed balance on a processor; runs fully offline |
| LFM2-700M (quantised, Q5 i believe) | Read the transcript, find names, map them to voices | Tiny language model built to run quickly on ordinary processors; ships tuned fast math routines (kernels) so inference is snappy even on a laptop |

The unifying rule of the project: every model must be small and must run on the processor. Bigger is not better here if it cannot finish in reasonable time on the machine you have.

## Data 
 **All of it comes from raw youtube and Apple Original content, simply to justify the usage of a pipeline over a crowded space**
 - One of the audio files is an mp3 which is an official open source meeting of an UK constituency meeting. 

---

## Cool bits, the drawbacks and "why" behind most of the things

Each of these is a problem we actually hit, and the answer we landed on. Kept short so they can be lifted straight into a call.

### Why Qwen was set aside for summarisation

**Problem.** We first used Qwen to produce meeting summaries. It writes noticeably better long-form prose — but it is heavy. Far more parameters mean slower loading, slower generation, more memory, and a sluggish feel on meeting-length recordings, all on an ordinary processor.

**Solution.** LFM2 is much smaller and is built for exactly this: it ships fast, tuned computation routines (kernels) for consumer processors, so each step is quick. For our actual work — listing names, choosing identities from a closed list, compact structured summaries — its quality is enough and the speed difference is dramatic. Both models speak the same text-in, structured-output interface, so the choice is a config switch, not a redesign: LFM2 by default for speed; switch back to Qwen when you want richer prose and can wait.

### The model kept labelling random speakers "Which" and "Road" for instance:

**Problem.** The small model, asked to list names, simply echoed every capitalised word it saw — so speakers ended up named "Which", "On", "Road", "Site". The old safety checks were useless: a capitalised English word almost always "occurs" in the transcript.

**Solution.** A name must be *used* like a name: spoken to as an address with a proper boundary ("Thank you, Chairman"), preceded by a title ("Mr Lee", "Councillor Perry"), or introduced ("my name is", "introduce", "hear from"). Common function words and interrogatives are rejected outright, and a name after a plain comma no longer counts — "At a stroke, flooding…" is a subject, not an addressee. Generic English rules only, nothing tailored to any one recording.

### A wrong name is worse than an honest unknown

**Problem.** Early on, a single weak guess from the model could attach a wrong name to a voice — and a confidently wrong transcript is worse than an anonymous one, because people trust what they read.

**Solution.** All evidence — who was addressed, who answered, who introduced themselves, the model's pick — is weighed, and a name is displayed only when the total clears a confidence bar. Below the bar the speaker keeps their generic label (Man 2, Speaker 4). The model is also only ever allowed to pick from a verified closed list, so it cannot invent anyone.

### The model confused who was being addressed

**Problem.** When one speaker said "Thank you, Dan", the model sometimes concluded that the *speaker* was Dan — grabbing the name of the person being talked to and giving it to the talker.

**Solution.** Address evidence now works against its own speaker: if a voice addresses a name, that voice cannot be that name. Instead the response-after-address pattern credits the name to the voice that answers next.

---

## Honest Limits

- Grokking system prompts to the language model makes things worse! it comes at a cost of cutting down on the model's context.
- Gender labelling is pitch-based and probabilistic; it is a helpful hint, never a claim.
- Names are only shown above a confidence threshold; silence is biased over a wrong name.
- Whisper is English-first; other languages will need a different transcription model, something along on the likes of Supertonic(open source but billion params), or this part can be replaced by a Sarvam/Eleven Labs API call. 
- Everything depends on audio quality: overlapping speech, distant microphones, and background noise degrade every stage above it.


---

## Where to Tune Things

- Everything comes down to VAD and Transcription, the better the model gets easier it is for the Language model to classify.
- While model intelligence on my mac wouldnt go beyond running the pipeline over LFM, Whisper and Silero, Hardware is the limit here. 
- Model files live in `models/`, recordings are expected in `audio_data/`, and finished transcripts are written to `outputs/`.
