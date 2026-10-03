# Elysium X 150 FR: research and design gate
Date: 3 October 2026. Status: design, not a trained model. No new metrics exist.

Build an Apache-2.0 Qwen2.5-1.5B-Instruct adapter for contextual emotion appraisal, with an explicit 150-coordinate product schema. NcR indexes co-active dimension subsets; it does not create training examples, prove psychological uniqueness, or imply state-of-the-art performance. Keep individual-speaker appraisal separate from the companion's response state and safety decisions.

## Data found (published counts, not yet downloaded/accepted counts)
| Source | Unit and count | Fit | Terms / decision |
|---|---|---|---|
| EmpatheticDialogues | ~25,000 conversations, English | emotional situations; short multi-turn | CC-BY-NC-4.0. Exclude unrestricted product training pending permission |
| DailyDialog | 13,118 dialogues, English | everyday multi-turn, emotion labels | CC-BY-NC-SA-4.0. Exclude unrestricted product training |
| MELD | 1,433 dialogues / 13,708 utterances (1,039/114/280 dialogues; 9,989/1,109/2,610 utterances) | multi-party sitcom, 7 emotions | repo GPL-3.0; underlying Friends rights not settled by code license. Quarantine pending review |
| EDiReF / E-MaSaC | 449 Hinglish dialogues / 11,908 utterances | 8 emotions, emotion-flip triggers, sitcom | author repository found, dataset rights not established. Quarantine |
| EmoInHindi | 1,814 dialogues / 44,247 utterances, 16 labels with intensity | Hindi, counselling/crime-victim scenarios, Wizard-of-Oz | agreement-gated; paper license is not dataset license. Exclude until access/terms established |
| IEMOCAP | approximately 12 hours; 10 actors | acted dyadic audio/text plus valence/activation/dominance | signed agreement, internal noncommercial research, restricted derivatives. Exclude |
| XED | English 17,530 emotional + 6,420 neutral unique lines; 31 projected languages reported in current repo | multilingual subtitle-level augmentation, NOT a ready conversation corpus | repo states CC-BY-4.0. Check underlying subtitle provenance, preserve attribution; never invent turn history |
| OASST1 | full release 161,443 messages / 66,497 trees / 35 languages; filtered HF splits 84,437 train + 4,401 validation message rows | real human assistant conversations; filter relevant branches | Apache-2.0 candidate. Trees/messages are not dialogue counts; no 150-label gold |
| UltraChat 200k | 207,865 train_sft + 23,110 test_sft dialogues; additional generation splits 256,032 + 28,304 | synthetic general conversation, not automatically intimate | MIT candidate, explicitly ChatGPT-generated. Teacher/origin terms review before unrestricted redistribution |

Do not add these counts together: they mix messages, lines, trees and dialogues, and include excluded sources. No verified all-language intimate corpus or human-labelled 150-axis corpus was found. Initial languages: English, Devanagari Hindi, Roman Hindi/Hinglish. Spanish, Arabic, French and other languages are separate expansion slices requiring quality review, not an 'all languages' claim.

## Exactly 150 dimensions: ten families of fifteen
These are operational text-appraisal coordinates, not 150 independent, validated human emotions. Each dimension needs a definition, examples, exclusion examples and annotator guidelines before annotation. Intensity is a value per dimension, not a duplicated low/medium/high label. A speaker can express multiple or conflicting dimensions. Most relational states must be unknown unless supported by dialogue evidence.

1. Positive affect: joy, contentment, amusement, excitement, delight, elation, serenity, relief, hope, optimism, gratitude, pride, admiration, awe, inspiration.
2. Distress and loss: sadness, grief, sorrow, loneliness, emptiness, disappointment, discouragement, despair, helplessness, hopelessness, regret, remorse, guilt, shame, embarrassment.
3. Threat and uncertainty: fear, anxiety, nervousness, apprehension, worry, dread, panic, insecurity, vulnerability, unease, suspicion, distrust, uncertainty, overwhelm, stress.
4. Opposition and injury: anger, annoyance, frustration, irritation, resentment, bitterness, indignation, outrage, disgust, contempt, envy, jealousy, humiliation, betrayal, hurt.
5. Bonding and care: affection, love, tenderness, compassion, empathy, caring, warmth, fondness, attachment, belonging, acceptance, trust, emotional_closeness, intimacy, solidarity.
6. Social appraisal: approval, disapproval, respect, disrespect, appreciation, feeling_seen, feeling_unheard, validation, invalidation, rejection, exclusion, social_comparison, status_threat, dignity, recognition.
7. Knowledge and surprise: curiosity, interest, confusion, surprise, astonishment, bewilderment, realization, understanding, doubt, skepticism, certainty, wonder, cognitive_conflict, disbelief, expectation_violation.
8. Agency and motivation: desire, longing, yearning, determination, resolve, motivation, ambition, confidence, self_efficacy, agency, powerlessness, perceived_control, autonomy, defiance, resignation.
9. Regulation and readiness: calmness, agitation, tension, restlessness, emotional_fatigue, composure, patience, impatience, defensiveness, openness, guardedness, emotional_numbness, emotional_recovery, willingness_to_engage, withdrawal.
10. Relational needs and repair: reassurance_seeking, support_seeking, comfort_seeking, connection_seeking, distance_seeking, boundary_assertion, boundary_discomfort, fear_of_abandonment, dependency_concern, reciprocity_expectation, forgiveness, reconciliation_desire, conflict_repair, relational_ambivalence, self_disclosure_readiness.

Some coordinates are emotional appraisals, relational needs or regulation states rather than basic emotion labels. Names that sound similar require contrast tests, not forced independent certainty. No diagnosis, inferred sexual consent, demographic prediction, personality scoring or safety authorization is derived from this vector.

## Model and state contract
For a target speaker and target turn, use ONLY preceding turns plus the current turn, preserving speaker IDs and roles. Output sparse JSON entries: dimension ID, strength [0,1], evidence turn IDs. An omitted dimension is not an observed negative label. Training records have a separate annotation mask and provenance per label: human / source-mapped / teacher / synthetic. Confidence is not the same as intensity; confidence calibration needs held-out observations.

The full vector x has 150 slots. Present/absent thresholds are dimension-specific and tuned on development data; unknown coordinates carry a mask. For active set A, r=|A|, there are C(150,r)=150!/(r!(150-r)!) possible subsets. C(150,2)=11,175; C(150,3)=551,300; all binary subsets including the empty set number 2^150=1,427,247,692,705,959,881,058,285,969,449,495,136,382,746,624. Compute these lazily, never enumerate the space. Invalid or unsupported combinations may be excluded by a versioned rule set.

A subset rank is unique ONLY for that selected subset and fixed ordered taxonomy. A state record must also retain taxonomy version, speaker, mask and strength vector; equal active subsets do not imply equal emotions. If quantizing strength for a deterministic state key, retain the unquantized vector and disclose collisions introduced by quantization. Use exact canonical serialization for identity, or a hash with normal collision caveats. NcR is the combinatorial index, not a learned psychological law.

## Data build
1. Pin source revision and preserve actual license/notice files. Download eligible sources only. Reconstruct OASST parent-child branches without mixing siblings, retaining original tree split. Filter topical relevance, language, PII and unsafe material. Count conversations, turns, labels and length tokens separately.
2. Create original synthetic scenarios with stratified languages, topics and lengths. Adult, non-explicit romance/intimacy, grief, reconciliation, support, boundaries, anger, joy, casual talk, sarcasm and safe distress. No scraped private chats or identifiable people. Plan 500-conversation pilot first; expansion toward 10k accepted conversations is a target, not an existing count.
3. Record provider, model/version, prompt hash, seed when supported, time, generation terms, language and synthetic=true. A teacher's labels are weak supervision, not human truth. Separate generation from independent critique. Include conflicting emotions, neutral/ambiguous inputs, corrections and history-dependent flips. Do not manufacture rich labels from a source with only 7 classes: other coordinates remain masked.
4. Deduplicate exact text and near-duplicate scenarios before train/dev/test. Split by original tree, scenario family, translation group and speaker template, never random overlapping windows. Translation is augmentation, not native human conversation. Keep all translations in one split.
5. Validate JSON/schema, speakers, chronological evidence, label IDs, finite strengths, no future leakage, language quality and token budget. Retain rejection counts and reasons. Review samples manually; no automatic filter establishes truth across all languages.

## Training candidate
Base: Qwen2.5-1.5B-Instruct (Apache-2.0). Qwen2.5-3B-Instruct has a NONCOMMERCIAL research license; do not substitute it. Start fresh from base rather than assuming X20 FR's Reddit adapter transfers. LoRA r16/r32 is a design candidate; select after a smoke test. QLoRA/gradient checkpointing, microbatch 1, gradient accumulation and an initial 2,048-token total budget are resource choices to test, not guaranteed T4 throughput. Sparse outputs avoid generating 150 bulky objects per turn.

One shared tokenizer/window implementation must govern train/eval/runtime: reserve answer budget, retain system/task contract and target turn, remove oldest complete prior turns only, explicitly flag context trimming, reject overlong targets rather than silently cutting labels or current input. Test long Hindi/Hinglish messages because token costs differ. Causal history only. Mask unknown labels; if training generative sparse JSON without an auxiliary masked classifier, document that its omission targets cannot be interpreted as fully observed negatives. Consider a 150-head masked classifier as a separate candidate, not an untested claim that JSON generation solves missing labels.

Checkpoint model, optimizer, scheduler, RNG, tokenizer, taxonomy, data hashes and run config to private Drive/HF. Colab free resources/GPU types are variable and not guaranteed. Verify the actual available runtime and storage before committing a long run. No paid API or GPU fallback without owner approval.

## Evaluation gates
Freeze test set before training. Published corpora evaluate ONLY their native dimensions or explicit documented many-to-one mapping. X20 FR is a baseline on common labels and same held-out conversations: compare latest-turn-only X20, base Qwen zero-shot with causal context, and new X150. Never compare its old GoEmotions F1 directly to new conversational F1.

Report strict JSON and schema validity on every attempt (invalid outputs count as failures), native-label micro/macro/per-label F1 with support, mapped-label F1, language slices, length/context slices, masked coordinate MAE, intensity agreement only against independently labelled intensity, Jaccard/state similarity over known labels and support. Gold unavailable = 'not measured', not dimension-wise accuracy. Show context ablation and counterfactual preceding turns. Report p50/p95 latency, hardware, batch size and token lengths. Use conversation-level bootstrap intervals and multiple seeds when feasible. Human review is required for a credible 150-coordinate gold evaluation; synthetic-held-out teacher agreement is not human emotional accuracy. State-of-the-art claims require fair published-benchmark comparisons, not a big state-space count or a low training loss.

## Sources
- https://github.com/facebookresearch/EmpatheticDialogues
- https://github.com/facebookresearch/EmpatheticDialogues/blob/main/LICENSE
- https://huggingface.co/datasets/roskoN/dailydialog
- https://github.com/declare-lab/MELD
- https://github.com/Helsinki-NLP/XED
- https://github.com/thu-coai/Emotional-Support-Conversation
- https://github.com/priyanshu-profile/EmoInHindi
- https://aclanthology.org/2022.lrec-1.627/
- https://github.com/LCS2-IIITD/EDiReF-SemEval2024
- https://arxiv.org/html/2402.18944v1
- https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf
- https://huggingface.co/datasets/OpenAssistant/oasst1/blob/main/README.md
- https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/blob/main/README.md
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE
- https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/LICENSE
- https://research.google.com/colaboratory/faq.html
Baseline source inspected via owner's connected GitHub: https://github.com/itsppm76/Elysium-X-20-FR (private). Its train/train.py confirms 200-token prompt cutting vs 220-token eval cutting; the new implementation must not copy this mismatch. README's reported metrics are historical baseline claims, not independently rerun X150 measurements.
