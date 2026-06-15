# AI/ML Intelligence Hackathon — Track Problem Statements

**Hackathon dates:** June 15–19, 2025 (submission) | June 20, 2025 (final presentation)
**Contact:** guru.aihackathon@gmail.com

---

## About the Dataset

All tracks use the same dataset: a financial transaction dataset modelled on
SAML-D data set.
The dataset contains approximately **1,000,000 transaction records** across
roughly **6,800 unique accounts**, with associated Know Your Customer (KYC)
records for each account.

The data is provided as:

- `augmented_saml_d.csv` — the main transaction table
- `ml_features.csv` — a pre-engineered feature table ready for ML use
- `party_registry.csv` — KYC records for all accounts
- `strs.xml` — Suspicious Transaction Reports (STRs) in XML format

Each track uses a subset of these files. You do not need to use all files for
your track. You are free to engineer additional features from the data.

> The dataset is synthetic and was built for research and educational purposes.
> It does not contain real personal information.

---

## Track 1 — Signal vs Noise Detection

### The problem

Financial institutions are required to file Suspicious Transaction Reports
(STRs) whenever they identify potentially suspicious activity.
However, the quality of these reports varies widely. Some reports are detailed and actionable; others are vague, generic, or provide little structured information. An analyst receives hundreds of reports and needs to
know which ones deserve immediate attention and which ones need to be sent back to the reporting entity for more detail.

Your task is to build a system that scores each STR report on how analytically complete and useful it is — producing a ranked list so analysts can prioritise their review queue.

### What you are given

- `strs.xml` — STR reports, each containing:
  - Structured coded fields (transaction mode, fund source/destination, account
    type, legal form, etc.)
  - A free-text `reason` narrative written by the reporting officer
  - Transaction and party details

### What you need to build

A model or scoring system that assigns each STR report a **completeness score
between 0 and 1**, where:

- **1.0** = the report gives the analyst a full, specific, actionable picture
- **0.0** = the report is so vague or generic that it tells the analyst almost nothing

Your system should be able to explain what makes a report high or low quality.

### What good looks like

- Your score ranking should separate clearly informative reports from vague
  ones
- You should be able to point to specific signals your model uses (e.g. how
  many coded fields are specific vs generic, whether the narrative contains
  specific amounts and dates, whether the customer explanation is present)
- A practical demo: given a low-scoring report, your system should be able to
  flag which aspects are weak.

---

## Track 2 — Entity Resolution & Record Linkage

### The problem

In financial crime investigations, the same person often holds accounts at
multiple banks under slightly different recorded details — a name abbreviated
at one branch, a birthdate entered differently at another, a citizenship number recorded with or without hyphens. Linking these records to the same underlying person is called entity resolution, and it is one of the most labour-intensive tasks for an analyst.

Your task is to build a system that identifies which accounts in the dataset
belong to the same real-world person, despite inconsistencies in how their
details were recorded.

### What you are given

- `party_registry.csv` — KYC records for all ~6,800 accounts, each containing:
  - Full name, date of birth, citizenship number, phone, address
  - Bank and branch where the account is held
  - Account type and opening date

### What you need to build

A system that **groups accounts by the real person behind them** — producing
clusters where each cluster contains all accounts belonging to one individual.
Accounts that have no duplicates in the dataset should each form their own
singleton cluster.

### What good looks like

- Your system correctly links accounts with name variants, formatting
  differences in citizenship numbers, or transposed birthdates to the same
  person
- It does not incorrectly merge accounts that belong to different people
- It scales — a solution that requires manual comparison of every pair is not
  practical at this data size

---

## Track 3 — Network & Graph Intelligence

### The problem

Money laundering rarely involves a single transaction. It works through
networks — chains of transfers, fan-out structures where one source feeds many accounts, or circular flows where money eventually returns to the origin. These patterns are invisible when you look at individual transactions, but they become visible when you model the data as a graph.

Your task is to build a system that models the transaction data as a network
and identifies accounts or groups of accounts that show suspicious structural
behaviour.

### What you are given

- `augmented_saml_d.csv` — the full transaction table, where each row is a
  directed money transfer from a sender account to a receiver account
- `party_registry.csv` — KYC records for all accounts (node attributes)
- `ml_features.csv` — pre-engineered features including cross-border flags,
  transaction timing, and amount ratios

### What you need to build

A graph-based analysis system that:

1. **Constructs the transaction graph** (accounts = nodes, transactions = edges
   weighted by amount and time)
2. **Identifies suspicious subgraphs or nodes** — accounts or groups that show
   structural patterns consistent with known money laundering typologies such
   as smurfing (fan-out/fan-in), layering (chains of transfers), or circular
   flows
3. **Produces a ranked list of suspicious accounts or subgraphs** with a
   supporting explanation of what structural feature made them suspicious

### What good looks like

- Your system finds multi-hop patterns that are not visible at the transaction
  level.
- You can explain _why_ a flagged structure is suspicious in structural terms
  (e.g. "one account sends to 14 different accounts within 24 hours, all of
  which forward funds to the same collector")
- Your results go beyond degree centrality — surface-level high-degree nodes
  are easy to find; interesting results require reasoning about flow direction,
  amount conservation, timing, and multi-hop paths.

---

## Track 4 — Risk Scoring & Prioritization

### The problem

An analyst cannot investigate every account equally. They need a risk score
— a number that reflects how likely an account is to be involved in money
laundering — so they can direct their attention to the highest-risk cases
first. Building this score from transaction patterns, KYC attributes, and
behavioural signals is a core applied ML problem in financial crime.

Your task is to build a model that assigns a risk score to each account and
produces a ranked priority list that an analyst could actually use.

### What you are given

- `ml_features.csv` — pre-engineered features per transaction including:
  - Transaction amount, type, currency, cross-border flag
  - KYC-derived features: PEP flag, sanctions hit, account age, account type
  - Behavioural features: transaction velocity, amount variance
  - `Is_laundering` label (use this as your training signal)
- `party_registry.csv` — full KYC records for additional feature engineering

### What you need to build

A model that:

1. **Predicts risk at the account level** — aggregate transaction-level
   signals into a per-account risk score
2. **Ranks accounts** from highest to lowest risk
3. **Explains the score** — what features drove the risk assessment for a
   given account

### What good looks like

- Your ranked list surfaces genuinely suspicious accounts near the top, not
  just high-value or high-frequency ones
- You can explain what drove the score for any given account
- Your evaluation goes beyond accuracy — think about precision at the top of
  the ranked list (does rank 1–50 actually contain high-risk accounts?)
- A practical demo: given an account ID, your system produces its risk score
  and a human-readable explanation.

---

## Track 5 — Pattern & Behavior Discovery

### The problem

Not all suspicious activity is labelled. New laundering typologies emerge
constantly, and an analyst cannot rely solely on known patterns. Unsupervised
learning can surface behavioural archetypes in the data — groups of accounts
that behave similarly — without needing labels. Some of these archetypes will
correspond to suspicious behaviour; some will correspond to legitimate
behaviour types. The analyst's job is to investigate the archetypes, not
individual accounts.

Your task is to discover distinct behavioural groups in the account population
using unsupervised methods, characterise each group, and assess which groups
warrant further investigation.

### What you are given

- `augmented_saml_d.csv` — full transaction history for all accounts
- `ml_features.csv` — pre-engineered features
- `party_registry.csv` — KYC attributes

You are deliberately **not** given account-level labels for this track. The
goal is discovery, not classification.

### What you need to build

A system that:

1. **Engineers behavioural features** at the account level from the transaction
   history (transaction cadence, amount distribution, counterparty patterns,
   cross-border behaviour, etc.)
2. **Clusters accounts** into behavioural archetypes using unsupervised methods
3. **Characterises each archetype** — what does a typical account in this
   cluster look like? What behaviour defines it?
4. **Flags archetypes of interest** — which clusters show characteristics
   consistent with suspicious activity, and why?

### What good looks like

- Your archetypes are interpretable and distinct — you can describe each one in
  plain language ("accounts that receive money from many sources and immediately
  forward it")
- The archetypes reflect actual behaviour differences, not just data artefacts
  (e.g. grouping by bank or currency alone is not meaningful)
- You can connect at least one archetype to a known money laundering pattern
  and explain the connection

### Hints

- Account-level behavioural features are more useful than raw transaction rows:
  compute things like flow ratio (total out / total in), counterparty churn,
  coefficient of variation of inter-transaction time, and proportion of
  same-day round-trips
- Graph-derived features (in-degree, out-degree, clustering coefficient) can
  enrich your clustering if you build the transaction graph
- The `Is_laundering` flag can be used _after_ clustering as a validation
  check — see whether your suspicious archetypes correlate with it — but do not
  use it during the clustering itself

---

## Track 6 — AI-Powered Analysis & Reporting

### The problem

When a bank or financial institution files an STR, the reporting officer writes
a free-text narrative explaining why the activity was flagged. These narratives
can be thousands of words long, describing account history, transaction
timelines, customer explanations, and the officer's suspicion. An analyst
who receives hundreds of reports cannot read each one in full. They need a
concise, accurate summary that preserves the key facts — amounts, parties,
dates, and the nature of the suspicion — so they can triage quickly.

Your task is to build an AI-powered summarisation system that takes a long
reporting-officer narrative and produces a short, accurate, analyst-facing
summary.

### What you are given

- `strs.xml` — STR reports, each containing a `reason` field with a
  1,000–8,000 character narrative written by the reporting officer
- `augmented_saml_d.csv` — the underlying transaction data (useful context for
  verifying factual accuracy)
- A sample of **gold summaries** for a subset of reports — human-written
  reference summaries you can use to evaluate your system (do not train on
  these directly)

### What you need to build

A summarisation system that:

1. **Takes a long STR narrative as input** (plus optionally the structured
   transaction fields)
2. **Produces a concise summary** of 100–200 words
3. **Preserves all key facts** — every amount, party name, date, and account
   number mentioned in the narrative must either appear in the summary or be
   explicitly excluded for a clear reason
4. **Is usable at scale** — it should be able to process all ~50,000 reports,
   not just one at a time

### What good looks like

- Summaries are factually faithful — no amounts invented, no parties dropped,
  no dates shifted
- Summaries are concise and structured — an analyst can read one in under 30
  seconds and know whether to investigate further
- Your system scores well on both fluency metrics (ROUGE-L, BERTScore) **and**
  factual faithfulness — a fluent summary that drops a key fact is not a good
  summary in this domain
- You can demonstrate your system on a live example during the final
  presentation.

---

## Submission requirements (all tracks)

Regardless of which track you choose, your submission must include:

1. **GitHub repository** — all source code, publicly accessible
2. **EDA notebook** — Jupyter or Google Colab, covering the dataset as it
   relates to your track
3. **Technical documentation** — explain your approach, design decisions, and
   results
4. **Presentation slides** — PPT format.

Refer to the main hackathon guidelines for full evaluation criteria and rules.
