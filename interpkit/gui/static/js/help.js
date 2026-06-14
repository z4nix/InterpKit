// Beginner-facing help content for every op, rendered as a collapsible
// card atop each panel (see components/help-panel.js).
//
// Mini-markup understood by the renderer:
//   **bold**        -> emphasis
//   `code`          -> inline code / module paths / field names
//   [label](op)     -> clickable link to another op panel (#/op/<op>)
//   \n              -> line break
//
// Each entry: { summary?, what, when, workflow: [..], example }

export const HELP = {
  scan: {
    summary: 'Start here — a one-click overview of the model on your text',
    what:
      'Runs several core analyses at once (direct logit attribution, logit lens, attention, and input attribution) on your input and surfaces the headline findings. Think of it as an automatic first pass.',
    when:
      'Use it right after loading a model, or whenever you want a quick read on how the model handles a prompt before diving into a specific tool.',
    workflow: [
      'Start here, then follow whatever stands out into its dedicated panel: [Logit lens](lens), [DLA](dla), [Attention](attention), or [Input attribution](attribute).',
      'Pick a **text** where the next-token prediction is interesting — a factual completion works well.',
    ],
    example:
      'Text: `The Eiffel Tower is in the city of` — scan shows which layers and heads drive the `Paris` prediction.',
  },

  inspect: {
    summary: 'See the architecture and the module paths you will type elsewhere',
    what:
      "Shows the model's detected architecture: family, number of layers, hidden size, attention heads, and the full module tree with the canonical paths interpkit uses to hook into it.",
    when:
      'Use it to learn the **module paths** that other tools ask for (the `at` / `module` fields), and to confirm interpkit identified the architecture correctly.',
    workflow: [
      'Copy a path from the **Blocks** table (e.g. `blocks.6`) to use as the target in [Steer](steer), [Activations](activations), [Ablate](ablate), or [DLA](dla).',
      'No inputs needed — just open it.',
    ],
    example:
      'For GPT-2 you will see 12 blocks named `blocks.0` … `blocks.11`, each with an attention and an MLP sub-module.',
  },

  report: {
    summary: 'Bundle the core analyses into one shareable HTML page',
    what:
      'Builds a single self-contained HTML report with the prediction, DLA, logit lens, attention, and attribution for one input.',
    when: 'Use it when you want a polished artifact to keep or share, rather than clicking through panels.',
    workflow: [
      'Same analyses as [Scan](scan), just packaged as a downloadable page.',
      'Set **save** to a file path to keep it; otherwise it renders inline here.',
    ],
    example: 'Text: `The capital of France is` → an HTML page you can open in any browser.',
  },

  lens: {
    summary: 'Watch the prediction form, layer by layer',
    what:
      "Takes the hidden state at every layer and decodes it as if it were the final output, so you can watch the prediction take shape. Each cell is the top-1 token that layer 'believes' comes next.",
    when:
      "Use it to find **where** in the network a prediction or concept emerges — the answer to 'at which layer does the model decide?'",
    workflow: [
      'The layer where the right token first appears is often the best place to intervene — a great way to choose the `at` layer for [Steer](steer), [Activations](activations), or [Patch](patch).',
      'Early layers look like noise or copying; the prediction usually crystallizes in the upper-middle layers. Hover any cell for its top-5 tokens.',
      'If early layers look noisy, train a [tuned lens](train-tuned-lens) and paste its path into the **tuned lens** field for cleaner readouts.',
    ],
    example:
      'Text: `The capital of France is` — watch `Paris` climb the ranks, and note the first layer where it reaches the top.',
  },

  dla: {
    summary: 'Attribute one prediction to specific heads and MLPs',
    what:
      'Breaks the final logit for a token into additive contributions from each component — every attention head, MLP, and the embeddings — so you see which parts pushed the prediction up or down.',
    when:
      "Use it to attribute a specific prediction to specific components: 'which heads or MLPs are responsible for this token?'",
    workflow: [
      'Positive bars push toward the token, negative bars push against it.',
      'Top-contributing layers are strong candidates to steer or ablate — feed them to [Steer](steer) or [Ablate](ablate).',
      'Set **token** to the word you care about (defaults to the predicted token); set **position** to analyze a different spot.',
      'Enable **sae** with an **sae_at** module to attribute through interpretable SAE features instead of raw heads.',
    ],
    example:
      'Text: `The capital of France is`, token: ` Paris` — see which upper-layer heads carry the fact.',
  },

  attribute: {
    summary: 'Which input words drove the prediction?',
    what:
      'Gradient-based saliency over the input tokens. Red tokens pushed toward the target prediction, blue tokens against it.',
    when: "Use it to answer 'which words in my prompt mattered?' — handy for spotting what the model keys on.",
    workflow: [
      'Complements [DLA](dla): DLA attributes to internal components, this attributes to the input tokens.',
      'Some architectures only give a reliable *ranking* of tokens, not exact magnitudes — the panel will say so when that is the case.',
      'Set **target** to attribute toward a specific token, and try different **method** options.',
    ],
    example:
      'Text: `The movie was absolutely fantastic and I` — watch `fantastic` light up as driving a positive continuation.',
  },

  attention: {
    summary: 'See how each head routes information between tokens',
    what:
      'For each attention head you pick, shows a heatmap of how much every token attends to every other token (rows = query, columns = key), plus the head entropy and its strongest token pairs.',
    when: "Use it to study information routing: 'which tokens is this head moving information between?'",
    workflow: [
      'Use the **layer** and **head** selectors above the heatmap to browse all heads.',
      'Low-entropy heads focus sharply on a few tokens (often the interesting ones); high-entropy heads spread attention broadly.',
      'Heads flagged by [DLA](dla) or [Find circuit](find-circuit) are the ones worth inspecting here.',
    ],
    example:
      'Text: `The cat sat on the mat because it was tired` — look for a head that links `it` back to `cat`.',
  },

  activations: {
    summary: 'Inspect the scale and statistics of a layer',
    what:
      'Captures summary statistics (mean, std, min, max, L2 norm, shape) of the activations at a module you choose, for a given input.',
    when:
      'Use it to check the magnitude and health of activations at a layer — useful before steering (to pick a sensible **scale**) or when debugging.',
    workflow: [
      'Get the module path from [Inspect](inspect), or type something like `blocks.6`.',
      'The L2 norm is the typical magnitude of the residual stream there — steering vectors should be scaled relative to it.',
    ],
    example: 'at: `blocks.6`, text: `Hello world` → statistics for the residual stream after block 6.',
  },

  trace: {
    summary: 'Causally localize where a behaviour lives',
    what:
      'Runs a clean prompt and a corrupted one, then patches clean activations into the corrupted run module-by-module to see which modules **restore** the correct answer — a causal importance ranking.',
    when:
      "Use it to causally localize a behaviour, not just correlate with it. The gold-standard 'which component matters?' test.",
    workflow: [
      'You need a **clean** and a **corrupted** prompt that differ minimally but change the answer (swap one key word).',
      'For a faster first look, run [ATP](atp) (approximate) before a full trace.',
      '**mode** can rank whole modules or give a layer × position heatmap.',
      'High-effect modules are prime [Patch](patch), [Ablate](ablate), or [Steer](steer) targets.',
    ],
    example:
      'clean: `The Eiffel Tower is in Paris`, corrupted: `The Colosseum is in Paris` — trace which modules carry `Eiffel → Paris`.',
  },

  patch: {
    summary: 'Confirm one module’s causal role',
    what:
      "Copies one module's activations from the clean run into the corrupted run and measures how much the output moves — a single, targeted causal test.",
    when:
      'Use it once you have a module in mind (from [Trace](trace), [DLA](dla), or [ATP](atp)) and want to confirm its causal role.',
    workflow: [
      'Same clean/corrupted setup as [Trace](trace), but for one **at** module instead of all of them.',
      'Narrow to a specific **head** or **positions** to localize further.',
    ],
    example: 'clean/corrupted as in Trace, at: `blocks.8.attn` — does patching just that attention layer restore the answer?',
  },

  ablate: {
    summary: 'What breaks if I remove this component?',
    what:
      "Removes a module's contribution — by zeroing it, replacing it with its mean, or resampling — and measures how the prediction changes.",
    when: 'Use it to confirm a component is *necessary* for a behaviour.',
    workflow: [
      '**method**: `zero` is the bluntest; `mean` and `resample` are gentler and often more faithful.',
      'A big prediction change means the module mattered; little change means it is redundant for this input.',
      'Pair with [DLA](dla) or [Trace](trace) to choose what to ablate.',
    ],
    example: 'at: `blocks.6.mlp`, method: `zero` — see how much that MLP mattered.',
  },

  decompose: {
    summary: 'Split the residual stream into its component contributions',
    what:
      'Breaks the residual stream at one position into the additive contributions of every component that has written to it so far (embeddings, each attention layer, each MLP).',
    when:
      "Use it to see what is 'in' the residual stream at a given token and layer — which components built up the current representation.",
    workflow: [
      'The residual stream is a sum of component outputs; this shows that sum pulled apart.',
      'Complements [DLA](dla) (which projects to the output) by looking at the stream itself.',
    ],
    example: 'text: `The capital of France is`, position: last token — see which layers wrote the most into the final residual.',
  },

  diff: {
    summary: 'Compare two models layer by layer',
    what:
      'Runs the same input through this model and a second model and compares their per-layer activations.',
    when:
      "Use it to study how fine-tuning or a different checkpoint changed the internals — 'what did training change, and where?'",
    workflow: [
      'Set **model_b** to another HuggingFace id with a compatible architecture (e.g. a base model vs its fine-tune).',
      'Large per-layer differences point to where the two models diverge.',
    ],
    example: 'model_b: `distilgpt2` against a loaded `gpt2`, text: `Hello` — compare the layer activations.',
  },

  probe: {
    summary: 'Find which layer represents a concept (and a direction to steer)',
    what:
      'Trains a simple linear classifier on the activations at a chosen module to test whether a concept is **linearly decodable** there. High accuracy means the concept is represented as a direction at that layer.',
    when: 'Use it to locate where a concept lives in the network, and to get a direction you can steer along.',
    workflow: [
      "This is one of the best ways to answer 'which layer should I steer at?' — sweep a few layers and steer at the one where the probe scores highest.",
      'Provide **texts** + **labels** (e.g. positive vs negative sentences), or a **data_path**.',
      'The probe weight vector *is* a steering direction; the contrastive vector in [Steer](steer) is a quick alternative.',
    ],
    example: 'at: `blocks.6`, texts: happy vs sad sentences, labels: 1/0 — accuracy near 1.0 means sentiment is linearly there.',
  },

  steer: {
    summary: 'Push the model along a direction and compare the effect',
    what:
      'Adds a direction to the residual stream at a chosen layer to nudge the behaviour, then shows the original vs steered next-token distributions side by side. The direction can be a **contrastive vector** (positive minus negative examples) or an **SAE feature**.',
    when:
      'Use it to causally test and control what a direction or feature does — make the model more positive, more formal, focused on a topic, and so on.',
    workflow: [
      '**Which layer (`at`) should I use?** Steering usually works best in the **middle layers**, where abstract concepts live — early layers are too tied to surface tokens, late layers are already committed to the output. For a 12-layer model, start around `blocks.6` and sweep a few.',
      '**Find the best layer empirically:** run a [Linear probe](probe) across layers and steer where it is most accurate; use [Logit lens](lens) to see where the concept emerges; or use [DLA](dla) to see which layers contribute to the target token.',
      'Get exact module paths from [Inspect](inspect), and check the typical activation magnitude with [Activations](activations) to choose a sensible **scale**.',
      '**scale** controls strength: too low does nothing, too high produces gibberish — sweep it.',
      'For SAE steering, set **sae**, a **feature** id, and **feature_mode** (`add` or `clamp`) instead of positive/negative. Find feature ids in [SAE features](features).',
    ],
    example:
      'Steer toward positive sentiment — at: `blocks.6`, positive: `I love this, it is wonderful`, negative: `I hate this, it is awful`, scale: `4`. Then raise the scale and watch the steered distribution shift, then carry the same settings into [Generate](generate).',
  },

  generate: {
    summary: 'Generate full text with an intervention active every step',
    what:
      'Generates text while keeping a steering vector, SAE feature, or ablation **active at every decode step** — so you see the effect on full continuations, not just the next token.',
    when: 'Use it after you have found a good direction and layer in [Steer](steer) and want to see it shape real generated text.',
    workflow: [
      'First dial in **at**, **positive**, **negative**, and **scale** (or an SAE **feature**) in [Steer](steer), then reuse them here.',
      '**sample** with **temperature** / **top_p** controls randomness; turn sampling off for deterministic greedy output.',
      'Set **ablate_at** to knock out a module during generation, instead of or alongside steering.',
      '**capture** records per-step internals you can inspect in the result.',
    ],
    example: 'prompt: `My day was`, at: `blocks.6`, positive/negative as in Steer, scale: `6` — then compare against scale `0`.',
  },

  chat: {
    summary: 'Converse with an instruct model (interventions apply)',
    what:
      "A conversation with an instruction-tuned model through its chat template. Any active interventions apply to the chat too.",
    when: 'Use it with instruct/chat models (e.g. SmolLM2-Instruct) to interact naturally and to test steering in dialogue.',
    workflow: [
      'Only works on models with a chat template — base models like GPT-2 will not behave as a chat model.',
      'Open **Chat settings** to set a system prompt, max tokens, and sampling.',
      'Enter sends; Shift+Enter makes a newline; Reset clears the thread.',
    ],
    example: 'System: `You are a terse pirate.` then say `Tell me about the weather.`',
  },

  features: {
    summary: 'Decode a layer into interpretable SAE features',
    what:
      'Passes the activations at a module through a **Sparse Autoencoder** to decompose them into a small set of interpretable features that were active — turning a dense vector into a readable list.',
    when:
      'Use it to interpret what the model represents at a layer in human terms, and to find feature ids to steer with.',
    workflow: [
      'You need an **sae** (a HuggingFace SAE id or path) trained for the **at** module.',
      'The top features here are the ids you can plug into [Steer](steer) / [Generate](generate) as `feature`.',
      'Use [Max-activating examples](maxact) to learn what a given feature *means* by seeing what fires it.',
    ],
    example: 'sae: an SAE id, at: `blocks.6`, text: `The Golden Gate Bridge` — see which features fire.',
  },

  'find-circuit': {
    summary: 'Automatically discover and verify a behaviour’s circuit',
    what:
      'Automated circuit discovery: searches for the **minimal set of components** that explains a behaviour and verifies causally that they are sufficient. The end-to-end version of manual tracing.',
    when: 'Use it when you want the whole circuit for a behaviour, not just one component.',
    workflow: [
      'Needs a **clean** / **corrupted** pair like [Trace](trace).',
      'Under the hood it uses edge attribution ([EAP](eap)) to propose, then prunes and verifies — run [ATP](atp) / [EAP](eap) first to understand the candidates.',
      '**threshold** trades circuit size against faithfulness.',
      'Long-running — the progress bar tracks it and you can cancel.',
    ],
    example:
      'clean: `John gave a drink to Mary`, corrupted: `John gave a drink to John` — discover the indirect-object-identification circuit.',
  },

  atp: {
    summary: 'Fast, approximate causal scores for every module',
    what:
      'Approximates a full causal trace with a first-order (gradient) estimate, scoring every module from just three passes. Fast, signed, approximate.',
    when: 'Use it as the quick first look before committing to a slow [Trace](trace) or [Find circuit](find-circuit).',
    workflow: [
      'Same clean/corrupted setup as [Trace](trace).',
      'Treat the scores as a ranked shortlist; confirm the top ones with an exact [Patch](patch) or [Trace](trace).',
    ],
    example: 'clean/corrupted IOI prompts → a ranked list of candidate modules in seconds.',
  },

  eap: {
    summary: 'Circuit discovery at edge granularity',
    what:
      'Like ATP but at **edge** granularity — scores every component-to-residual edge with integrated gradients, so you see not just which nodes matter but how they connect.',
    when: 'Use it for finer-grained circuit discovery, or as the candidate generator behind [Find circuit](find-circuit).',
    workflow: [
      '**ig_steps** sets the integrated-gradients resolution (higher = more accurate, slower).',
      '**top_k_edges** limits how many edges to report. The top edges define the circuit wiring.',
    ],
    example: 'clean/corrupted IOI prompts → top edges such as `head L9H9 → logits`.',
  },

  maxact: {
    summary: 'Find what a neuron, feature, or head actually detects',
    what:
      'Scans a corpus for the examples that most strongly activate a specific unit — a neuron, an SAE feature, or an attention head — and shows them with per-token activation coloring. The standard way to *name* what a unit does.',
    when: "Use it to interpret a unit: 'what does this neuron / feature respond to?'",
    workflow: [
      'Specify the unit with **at** plus one of **neuron**, **feature** (with an **sae**), or **head**.',
      'Provide a corpus via **texts**, **texts_path**, or a **dataset** name.',
      'Pair with [SAE features](features): find a feature there, then explain it here.',
      'Long-running over big corpora — the progress bar shows examples scanned.',
    ],
    example: 'at: `blocks.6.mlp`, neuron: `1420`, dataset: a text corpus → the sentences that light it up reveal its concept.',
  },

  'train-tuned-lens': {
    summary: 'Train a cleaner lens for early-layer readouts',
    what:
      "Trains small per-layer affine maps (a 'tuned lens', Belrose et al. 2023) that translate each layer's hidden state into the output basis, giving cleaner early-layer readouts than the raw logit lens.",
    when: 'Use it once per model when the plain [Logit lens](lens) looks noisy in early layers, then reuse the saved lens.',
    workflow: [
      'Provide a small **corpus** (or **corpus_path**) to train on; a few hundred steps usually suffices.',
      'It saves to a path — paste that path into the **tuned lens** field of [Logit lens](lens) to use it.',
      'Long-running (it trains a network); the progress bar tracks steps and you can cancel.',
    ],
    example: 'corpus: a few paragraphs, steps: `250` → save, then open [Logit lens](lens) with the saved path.',
  },
};
