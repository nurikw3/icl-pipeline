# Graph-aware ICL retrieval: detailed explanation with real run data

This document explains the graph-aware in-context learning retrieval that was
added to the project. It uses real data from the latest run:

```text
outputs/predictions_en_graph_common-graph_ppr-hybrid_graph_k8_graph_openai_gpt-oss-120b.csv
outputs/prompts_en_graph_common-graph_ppr-hybrid_graph_k8_graph_openai_gpt-oss-120b.jsonl
outputs/retrieved_examples_en_graph_common-graph_ppr-hybrid_graph_k8_graph_openai_gpt-oss-120b.jsonl
```

Run configuration:

```text
target_lang       = en
source_lang       = chg
retrieval_backend = graph
strategies        = graph_common, graph_ppr, hybrid_graph
k                 = 8
sample_fraction   = 0.1
only_sentences    = true
train rows        = 66
test rows         = 14
model             = openai/gpt-oss-120b
```

The important point: the graph does not translate. The graph only selects the
`k=8` demonstration examples that are inserted into the ICL prompt. Translation
is still performed by the LLM.

```text
test source_text -> graph retrieval -> 8 train examples -> prompt -> LLM
```

## Кратко: какие подходы сравниваются

Все подходы решают одну и ту же задачу: выбрать `k=8` train examples для
in-context prompt. LLM, test set и prompt template остаются одинаковыми;
меняется только retriever.

| Подход | Что делает |
|---|---|
| BGE-M3 dense similarity | Кодирует `source_text` в dense embeddings и выбирает ближайшие train-примеры по cosine similarity. |
| BM25 similarity | Лексический baseline: выбирает примеры по совпадению токенов и BM25 score. |
| graph_common | Строит двудольный граф `train example <-> feature` и выбирает примеры по прямому weighted feature overlap. |
| graph_ppr | На том же графе запускает Personalized PageRank: probability mass идет от query features к train examples и обратно. |
| hybrid_graph | Смешивает dense similarity, graph_ppr и graph_common: `0.55*dense + 0.30*ppr + 0.15*common`. |

В prompt попадают только выбранные пары:

```text
Chagatai: <source_text>
English: <target_text>
```

Graph features, PageRank probabilities and dense scores are used only for
ranking before prompt construction; they are not inserted into the prompt.

## Code locations

The graph retriever is implemented in:

```text
retrieval.py
```

Main class:

```python
class GraphICLRetriever:
    ...
```

The pipeline creates it in:

```text
run_baseline.py
```

when:

```bash
--retrieval_backend graph
```

is used.

## What the graph is

The graph is a weighted bipartite graph:

```text
source nodes <-> feature nodes
```

There are no direct `source -> source` edges. Two source examples are close only
if they share feature nodes or are connected through feature-source-feature
walks.

### Source nodes

Each row from the filtered train set becomes one source node.

Example source node:

```text
source: tuz aççıqdur
target: salt is bitter
type: phrase
length: 2
sheet: Ch 1
```

### Feature nodes

For each source row, the code extracts graph features from `source_text` and
metadata. Feature extraction is implemented in `_row_features`.

For a source text such as:

```text
tuz aççıqdur
```

the graph can create features such as:

```text
token:tuz
token:aççıqdur
prefix3:tuz
prefix3:açç
suffix3:dur
suffix4:qdur
bigram:tuz aççıqdur
type:phrase
length:2
sheet:Ch 1
```

The current manually assigned base weights are:

| Feature kind | Base weight | Meaning |
|---|---:|---|
| token | 3.0 | Exact lexical overlap is very important. |
| bigram | 2.2 | Local phrase pattern is important. |
| suffix3 | 1.2 | Captures endings such as `dur`. |
| suffix4 | 0.9 | A slightly longer ending signal. |
| prefix3 | 0.6 | Weak lexical/morphological signal. |
| type | 0.8 | `word`, `phrase`, `sentence`. |
| length bucket | 0.6 | Similar source length. |
| sheet | 0.4 | Same textbook chapter/section. |

This is a deliberately simple first version. It is not a learned graph model.
It is a hand-built weighted feature graph.

## IDF weighting

The base weight is not the final edge weight. Every feature is also multiplied
by an IDF factor:

```text
idf(f) = log((N + 1) / (df(f) + 1)) + 1
```

where:

```text
N     = number of train source nodes
df(f) = number of source nodes connected to feature f
```

The final source-feature edge weight is:

```text
w(source, feature) = base_weight(feature) * idf(feature)
```

This means rare features are stronger than common features. For example,
`type:phrase` appears in many rows and therefore receives less discriminative
power. A rarer token or suffix receives more power.

## Real test example

The real `test_index=0` from the saved run is:

```text
id:          Ch 1_12_en
sheet:       Ch 1
source_text: şorpa göşt wa tuzdur
reference:   soup is meat and salt
type:        phrase
length:      4
```

The LLM predictions for this same test item were:

| Strategy | Prediction |
|---|---|
| graph_common | The soup is meat and salty. |
| graph_ppr | Soup is meat and salt. |
| hybrid_graph | Soup is meat and salt. |

## Real query features

For:

```text
şorpa göşt wa tuzdur
```

the raw query features are:

| Feature | Base weight |
|---|---:|
| token:şorpa | 3.0 |
| token:göşt | 3.0 |
| token:wa | 3.0 |
| token:tuzdur | 3.0 |
| bigram:şorpa göşt | 2.2 |
| bigram:göşt wa | 2.2 |
| bigram:wa tuzdur | 2.2 |
| suffix3:rpa | 1.2 |
| suffix3:öşt | 1.2 |
| suffix3:dur | 1.2 |
| suffix4:orpa | 0.9 |
| suffix4:göşt | 0.9 |
| suffix4:zdur | 0.9 |
| prefix3:şor | 0.6 |
| prefix3:göş | 0.6 |
| prefix3:tuz | 0.6 |
| type:phrase | 0.8 |
| length:3-4 | 0.6 |
| sheet:Ch 1 | 0.4 |

Only features that also exist in the train graph can participate in graph
retrieval. For this query, the seen weighted features are:

| Feature | Weighted value | Restart probability | Train df |
|---|---:|---:|---:|
| token:wa | 10.238799 | 0.503389 | 5 |
| suffix3:dur | 3.750301 | 0.184383 | 7 |
| prefix3:tuz | 2.463648 | 0.121125 | 2 |
| sheet:Ch 1 | 1.438102 | 0.070704 | 4 |
| type:phrase | 1.232905 | 0.060616 | 38 |
| length:3-4 | 1.215983 | 0.059784 | 23 |

The restart probabilities are computed by normalizing the seen weighted query
features:

```text
P_restart(f) = w_query(f) / sum_seen_features w_query(f)
```

For example:

```text
P_restart(token:wa) = 10.238799 / 20.339739 = 0.503389
```

This is the origin of the "probability" in `graph_ppr`. It is not LLM
confidence. It is normalized probability mass for a random walk over graph
features.

## Strategy 1: graph_common

`graph_common` is direct weighted feature overlap.

For query `q` and source example `s`, both are represented as sparse weighted
feature vectors:

```text
v_q[f] = query feature weight
v_s[f] = source feature weight
```

The score is cosine similarity over shared graph features:

```text
score_common(q, s) =
    sum_f v_q[f] * v_s[f]
    /
    (||v_q|| * ||v_s||)
```

Only shared features contribute to the numerator.

For the real query:

```text
şorpa göşt wa tuzdur
```

`graph_common` selected these 8 examples:

| Rank | Source | Target | Score |
|---:|---|---|---:|
| 1 | ékinci mäs̱näwiyāt wa qaṣāyıd wa ǧäzaliyāt wa muqaṭṭaʾāt wa rubāʾiyāt wa barça aşʿārnı fahmlemeklik ʿArab wa Fārs wa Türk luǧatlarınıñ maʾnāsını bilmek | The second is understanding mathnawis qasidas ghazals and short poems and all kinds of poetry and knowing Arabic Farsi and Turki | 0.519089 |
| 2 | anıñ dostı wa barça ādam ferzendlerige wa cinlerge ibergen élçisi ulū‘l-ʿazm wa risālat wa nubūwwat wa ḫātimat bu tört märtebeni ʿināyat qılıp bergen rasūlı Muḥammad muṣṭafānıñ durūdındın soñ | after praising the messenger Muhammad the chosen one His friend and the emissary He sent unto all the children of Adam and to the djinn who held the four stations of the decision the bringing of the message the prophecy and the Seal | 0.382477 |
| 3 | äwwel sipāhgerlikniñ qānūnı wa yosunı kim neçük atlanmaq wa yürümek wa yawǧa yasaw yasamaq | The first is the law and manner of the military profession including how to ride and how to march and how to array a ferocious army | 0.339290 |
| 4 | wa ǧarı̇̄bdin biri budur | This is one of the stranger tales | 0.243554 |
| 5 | Samarqand wa Ḫocand bolǧay | "it ought to be Samarqand and Khujand" | 0.236580 |
| 6 | tuz aççıqdur | salt is bitter | 0.079030 |
| 7 | Qoy soyadur | He slaughters a sheep | 0.050166 |
| 8 | aq süt arzāndur | The white milk is cheap. | 0.049948 |

Why are long sentences ranked highly? Because the query contains `wa`, and the
sampled train graph has only 66 rows. In this small 10 percent subset, examples
with `wa`, matching length/type/sheet signals, and other shared graph features
can dominate. This is exactly why `hybrid_graph` is useful: it balances graph
signals with dense semantic similarity.

## Strategy 2: graph_ppr

`graph_ppr` is a Personalized PageRank-style random walk over the bipartite
graph.

Definitions:

```text
F = feature nodes
S = source nodes
q = restart distribution over query feature nodes
alpha = restart probability = 0.35
```

Transition from a feature node to source nodes:

```text
P(s | f) = w(s, f) / sum_s' w(s', f)
```

Transition from a source node back to feature nodes:

```text
P(f | s) = w(s, f) / sum_f' w(s, f')
```

At each iteration:

```text
p_S(t)     = p_F(t) * P(S | F)
p_F(t + 1) = (1 - alpha) * p_S(t) * P(F | S) + alpha * q
```

In code:

```python
ppr_steps = 8
restart_prob = 0.35
```

The final retrieval score is:

```text
score_ppr(s) = final probability mass assigned to source node s
```

For the real query, `graph_ppr` selected:

| Rank | Source | Target | Score |
|---:|---|---|---:|
| 1 | ékinci mäs̱näwiyāt wa qaṣāyıd wa ǧäzaliyāt wa muqaṭṭaʾāt wa rubāʾiyāt wa barça aşʿārnı fahmlemeklik ʿArab wa Fārs wa Türk luǧatlarınıñ maʾnāsını bilmek | The second is understanding mathnawis qasidas ghazals and short poems and all kinds of poetry and knowing Arabic Farsi and Turki | 0.189411 |
| 2 | anıñ dostı wa barça ādam ferzendlerige wa cinlerge ibergen élçisi ulū‘l-ʿazm wa risālat wa nubūwwat wa ḫātimat bu tört märtebeni ʿināyat qılıp bergen rasūlı Muḥammad muṣṭafānıñ durūdındın soñ | after praising the messenger Muhammad the chosen one His friend and the emissary He sent unto all the children of Adam and to the djinn who held the four stations of the decision the bringing of the message the prophecy and the Seal | 0.135841 |
| 3 | äwwel sipāhgerlikniñ qānūnı wa yosunı kim neçük atlanmaq wa yürümek wa yawǧa yasaw yasamaq | The first is the law and manner of the military profession including how to ride and how to march and how to array a ferocious army | 0.085763 |
| 4 | tuz aççıqdur | salt is bitter | 0.078828 |
| 5 | tuz şı̇̄rı̇̄n emes | salt is not sweet | 0.072947 |
| 6 | wa ǧarı̇̄bdin biri budur | This is one of the stranger tales | 0.052355 |
| 7 | aq süt arzāndur | The white milk is cheap. | 0.046520 |
| 8 | Samarqand wa Ḫocand bolǧay | "it ought to be Samarqand and Khujand" | 0.031644 |

The difference from `graph_common` is that PPR can reward examples not only for
direct feature overlap, but also for being in the same graph neighborhood after
several feature-source-feature propagation steps.

## Strategy 3: hybrid_graph

`hybrid_graph` mixes three signals:

```text
final_score =
    0.55 * dense_similarity
  + 0.30 * graph_ppr
  + 0.15 * graph_common
```

Before mixing, every score array is min-max normalized to `[0, 1]`.

The three components mean:

| Component | Meaning |
|---|---|
| dense_similarity | SentenceTransformer vector similarity over `source_text`. |
| graph_ppr | Random-walk structural relevance in the feature graph. |
| graph_common | Direct graph-feature overlap. |

For the real query, the top hybrid candidates and component scores were:

| Rank | Source | Dense | PPR | Common | Hybrid |
|---:|---|---:|---:|---:|---:|
| 1 | tuz aççıqdur | 1.0000 | 0.4151 | 0.1522 | 0.6974 |
| 2 | ékinci mäs̱näwiyāt wa qaṣāyıd wa ǧäzaliyāt wa muqaṭṭaʾāt wa rubāʾiyāt wa barça aşʿārnı fahmlemeklik ʿArab wa Fārs wa Türk luǧatlarınıñ maʾnāsını bilmek | 0.3358 | 1.0000 | 1.0000 | 0.6347 |
| 3 | anıñ dostı wa barça ādam ferzendlerige wa cinlerge ibergen élçisi ulū‘l-ʿazm wa risālat wa nubūwwat wa ḫātimat bu tört märtebeni ʿināyat qılıp bergen rasūlı Muḥammad muṣṭafānıñ durūdındın soñ | 0.4164 | 0.7167 | 0.7368 | 0.5545 |
| 4 | tuz şı̇̄rı̇̄n emes | 0.7413 | 0.3840 | 0.0460 | 0.5298 |
| 5 | Samarqand wa Ḫocand bolǧay | 0.6685 | 0.1656 | 0.4558 | 0.4857 |
| 6 | Qoy soyadur | 0.7660 | 0.1395 | 0.0966 | 0.4777 |
| 7 | aq süt arzāndur | 0.6671 | 0.2443 | 0.0962 | 0.4546 |
| 8 | murç qızıl emes kökdür | 0.7530 | 0.1230 | 0.0216 | 0.4543 |

This explains the behavior seen in the run: `hybrid_graph` moves `tuz aççıqdur`
to rank 1 because dense similarity strongly recognizes the lexical/semantic
proximity between `tuzdur` and `tuz`. The graph-only strategies rank long
sentences high because those sentences share strong graph features such as `wa`
and related structural signals in the small sampled graph.

## Real prompt: graph_common

This is the exact prompt saved for `test_index=0` and `strategy=graph_common`.

```text
Task: Translate the following Chagatai texts to English.
Output only the translation. Do not explain anything.

Examples:
Chagatai: ékinci mäs̱näwiyāt wa qaṣāyıd wa ǧäzaliyāt wa muqaṭṭaʾāt wa rubāʾiyāt wa barça aşʿārnı fahmlemeklik ʿArab wa Fārs wa Türk luǧatlarınıñ maʾnāsını bilmek
English: The second is understanding mathnawis qasidas ghazals and short poems and all kinds of poetry and knowing Arabic Farsi and Turki

Chagatai: anıñ dostı wa barça ādam ferzendlerige wa cinlerge ibergen élçisi ulū‘l-ʿazm wa risālat wa nubūwwat wa ḫātimat bu tört märtebeni ʿināyat qılıp bergen rasūlı Muḥammad muṣṭafānıñ durūdındın soñ
English: after praising the messenger Muhammad the chosen one His friend and the emissary He sent unto all the children of Adam and to the djinn who held the four stations of the decision the bringing of the message the prophecy and the Seal

Chagatai: äwwel sipāhgerlikniñ qānūnı wa yosunı kim neçük atlanmaq wa yürümek wa yawǧa yasaw yasamaq
English: The first is the law and manner of the military profession including how to ride and how to march and how to array a ferocious army

Chagatai: wa ǧarı̇̄bdin biri budur
English: This is one of the stranger tales

Chagatai: Samarqand wa Ḫocand bolǧay
English: "it ought to be Samarqand and Khujand"

Chagatai: tuz aççıqdur
English: salt is bitter

Chagatai: Qoy soyadur
English: He slaughters a sheep

Chagatai: aq süt arzāndur
English: The white milk is cheap.

Now translate:
Chagatai: şorpa göşt wa tuzdur
English:
```

LLM output:

```text
The soup is meat and salty.
```

Reference:

```text
soup is meat and salt
```

## Real prompt: graph_ppr

This is the exact prompt saved for `test_index=0` and `strategy=graph_ppr`.

```text
Task: Translate the following Chagatai texts to English.
Output only the translation. Do not explain anything.

Examples:
Chagatai: ékinci mäs̱näwiyāt wa qaṣāyıd wa ǧäzaliyāt wa muqaṭṭaʾāt wa rubāʾiyāt wa barça aşʿārnı fahmlemeklik ʿArab wa Fārs wa Türk luǧatlarınıñ maʾnāsını bilmek
English: The second is understanding mathnawis qasidas ghazals and short poems and all kinds of poetry and knowing Arabic Farsi and Turki

Chagatai: anıñ dostı wa barça ādam ferzendlerige wa cinlerge ibergen élçisi ulū‘l-ʿazm wa risālat wa nubūwwat wa ḫātimat bu tört märtebeni ʿināyat qılıp bergen rasūlı Muḥammad muṣṭafānıñ durūdındın soñ
English: after praising the messenger Muhammad the chosen one His friend and the emissary He sent unto all the children of Adam and to the djinn who held the four stations of the decision the bringing of the message the prophecy and the Seal

Chagatai: äwwel sipāhgerlikniñ qānūnı wa yosunı kim neçük atlanmaq wa yürümek wa yawǧa yasaw yasamaq
English: The first is the law and manner of the military profession including how to ride and how to march and how to array a ferocious army

Chagatai: tuz aççıqdur
English: salt is bitter

Chagatai: tuz şı̇̄rı̇̄n emes
English: salt is not sweet

Chagatai: wa ǧarı̇̄bdin biri budur
English: This is one of the stranger tales

Chagatai: aq süt arzāndur
English: The white milk is cheap.

Chagatai: Samarqand wa Ḫocand bolǧay
English: "it ought to be Samarqand and Khujand"

Now translate:
Chagatai: şorpa göşt wa tuzdur
English:
```

LLM output:

```text
Soup is meat and salt.
```

Reference:

```text
soup is meat and salt
```

## Real prompt: hybrid_graph

This is the exact prompt saved for `test_index=0` and `strategy=hybrid_graph`.

```text
Task: Translate the following Chagatai texts to English.
Output only the translation. Do not explain anything.

Examples:
Chagatai: tuz aççıqdur
English: salt is bitter

Chagatai: ékinci mäs̱näwiyāt wa qaṣāyıd wa ǧäzaliyāt wa muqaṭṭaʾāt wa rubāʾiyāt wa barça aşʿārnı fahmlemeklik ʿArab wa Fārs wa Türk luǧatlarınıñ maʾnāsını bilmek
English: The second is understanding mathnawis qasidas ghazals and short poems and all kinds of poetry and knowing Arabic Farsi and Turki

Chagatai: anıñ dostı wa barça ādam ferzendlerige wa cinlerge ibergen élçisi ulū‘l-ʿazm wa risālat wa nubūwwat wa ḫātimat bu tört märtebeni ʿināyat qılıp bergen rasūlı Muḥammad muṣṭafānıñ durūdındın soñ
English: after praising the messenger Muhammad the chosen one His friend and the emissary He sent unto all the children of Adam and to the djinn who held the four stations of the decision the bringing of the message the prophecy and the Seal

Chagatai: tuz şı̇̄rı̇̄n emes
English: salt is not sweet

Chagatai: Samarqand wa Ḫocand bolǧay
English: "it ought to be Samarqand and Khujand"

Chagatai: Qoy soyadur
English: He slaughters a sheep

Chagatai: aq süt arzāndur
English: The white milk is cheap.

Chagatai: murç qızıl emes kökdür
English: pepper is not red it is green

Now translate:
Chagatai: şorpa göşt wa tuzdur
English:
```

LLM output:

```text
Soup is meat and salt.
```

Reference:

```text
soup is meat and salt
```

## Reproducible code example

This snippet reconstructs the same 10 percent train/test sample and prints the
retrieved examples for the first test item.

```python
import pandas as pd

from run_baseline import filter_data, sample_dataframes
from retrieval import GraphICLRetriever

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_df, test_df = filter_data(
    train_df=train_df,
    test_df=test_df,
    target_lang="en",
    only_sentences=True,
)

train_df, test_df = sample_dataframes(
    train_df=train_df,
    test_df=test_df,
    fraction=0.1,
    seed=42,
)

row = test_df.iloc[0]
query = row["source_text"]
query_metadata = row.to_dict()

retriever = GraphICLRetriever(train_df=train_df)

for strategy in ["graph_common", "graph_ppr", "hybrid_graph"]:
    examples = retriever.retrieve(
        query=query,
        strategy=strategy,
        k=8,
        query_metadata=query_metadata,
    )

    print("\\n===", strategy, "===")
    print(examples[["source_text", "target_text", "type", "retrieval_score"]])
```

Run it with:

```bash
uv run python graph_debug.py
```

## Interpretation for defense

The graph backend can be described as:

> A lightweight heterogeneous feature graph for in-context example retrieval.
> The method represents each train source sentence as a source node connected
> to lexical, morphological, structural, and metadata feature nodes. Retrieval
> is performed either by direct weighted feature overlap, by a personalized
> PageRank-style random walk over the source-feature graph, or by a hybrid score
> that combines graph-based relevance with dense semantic similarity.

The probability in `graph_ppr` is:

> The probability mass of a random walk over the weighted bipartite graph,
> initialized from normalized query feature weights and repeatedly restarted
> toward the query feature distribution.

It is not:

```text
LLM confidence
translation probability
learned probability from a supervised graph model
```

It is:

```text
graph-structural relevance score
```

## Comparison results on the 10 percent smoke test

All rows below use:

```text
model       = openai/gpt-oss-120b
target_lang = en
k           = 8
n           = 14 test examples
```

| Retriever | Strategy | BLEU | chrF | BERTScore | Format error rate | Empty predictions | n |
|---|---|---:|---:|---:|---:|---:|---:|
| BGE-M3 dense | similarity | 10.294996 | 35.877459 | 79.059166 | 7.142857 | 0 | 14 |
| BM25 lexical | similarity | 6.772249 | 34.880287 | 80.980653 | 7.142857 | 0 | 14 |
| Graph | graph_common | 11.572381 | 35.245552 | 80.761707 | 0.000000 | 0 | 14 |
| Graph | graph_ppr | 11.749119 | 31.869322 | 79.799539 | 7.142857 | 0 | 14 |
| Graph + dense | hybrid_graph | 15.022431 | 36.727770 | 81.021279 | 0.000000 | 0 | 14 |

Best row in this comparison:

```text
hybrid_graph
```

It has the highest BLEU, chrF and BERTScore, and no format errors. This suggests
that the graph signal alone is useful but unstable on a small 10 percent sample,
while the hybrid retriever benefits from combining semantic embedding similarity
with explicit graph structure.

Additional user-provided row from another `intfloat`/graph_common run:

| Retriever | Strategy | BLEU | chrF | BERTScore | Format error rate | Empty predictions | n |
|---|---|---:|---:|---:|---:|---:|---:|
| Intfloat/E5 graph run | graph_common | 2.451427 | 17.452680 | 80.151719 | 0.000000 | 7 | 14 |

This row is not identical to the saved graph metrics above because it contains
`7` empty predictions. It should be treated as a separate run/configuration
until its manifest and predictions file are matched exactly.
