# Case Studies

This page illustrates how TRACE differs from representative baselines in audit-oriented latent risk discovery and verifiable response generation.

> **Note.** The Audit-OCIQA benchmark used in the paper is not publicly released, so its cases cannot be reproduced here. Both cases below are built on the **synthetic demo dataset** in [`datasets-public/audit-ociqa-demo/`](datasets-public/audit-ociqa-demo/); every department and amount is fabricated. The model outputs are **illustrative**: they show, on demo inputs, the qualitative behaviours we observed on Audit-OCIQA (over-inferred causes, misread numbers, downplayed risks, and conservative abstention). They are not logged outputs of actual runs. To obtain real outputs on the demo data, run `bash scripts/run_main_experiments.sh`.

All monetary values are in ten-thousand RMB. Queries and evidence are in Chinese in the dataset; English translations are shown below.

---

## Case 1: Latent Risk Discovery from Prior-Year Carryover Funds

Demo row: `R000010-5acec9ae` (Department D02, FY2024). Gold evidence: revenue and expenditure summary (`…-0002`), fiscal appropriation summary (`…-0003`), project expenditure table (`…-0006`), and budget explanation (`…-0008`) of document `bb5843317d`.

### Query

Please explain how much Department D02's 2024 project expenditure depends on prior-year carryover funds. The revenue and expenditure summary reports prior-year carryover funds of 252.50. Total expenditure in the fiscal appropriation summary is 11,208.12, which is 252.50 lower than the overall expenditure total of 11,460.62. Using the project expenditure table, please explain which projects the carryover funds are used for, how they were formed, and whether funding for these projects in future years is sustainable.

### Reference Answer

**Observation.** The 2024 revenue and expenditure summary reports prior-year carryover funds of 252.50 and total revenue of 11,460.62.

**Observation.** Total expenditure in the fiscal appropriation summary is 11,208.12, which is 252.50 lower than the overall expenditure total. This difference equals the prior-year carryover funds.

**Observation.** In the project expenditure table, unit funds from prior-year carryover total 252.50 and are used for equipment purchase and renewal and infrastructure maintenance.

**Cause.** The materials do not explain how the carryover funds were formed, for example whether they came from one-off items or accumulated balances.

**Impact.** The materials do not explain whether this reliance on carryover funds is sustainable, so the funding source for similar projects in later years is uncertain.

### Model Outputs (illustrative)

#### Closed-book Generation

**Observation.** In Department D02's 2024 health expenditure, 252.50 comes from prior-year carryover funds, which matches the difference between the fiscal appropriation summary and the revenue and expenditure summary.

**Cause.** The carryover funds most likely come from project funds that were not fully executed in 2023 because of procurement delays or cross-year construction cycles.

**Cause.** The current-year appropriation did not fully cover project needs, so the department used carryover balances to keep projects running, which reflects imprecise budget preparation.

**Impact.** Relying on carryover funds weakens the guiding role of the annual budget. If no carryover is available next year, equipment and maintenance projects may be interrupted.

#### BM25-RAG

**Observation.** Of Department D02's 2024 health expenditure of 11,460.62, 252.50 comes from prior-year carryover funds rather than current-year appropriation.

**Cause.** The 252.50 consists of unspent 2023 balances of equipment purchase and infrastructure maintenance projects that were rolled over into 2024.

**Impact.** Because these are one-off historical balances, they are not sustainable. Without new carryover or extra appropriation, similar projects will face a funding gap.

#### Naïve Dense RAG

**Observation.** In Department D02's 2024 project expenditure, 252.50 comes from prior-year carryover funds and is not covered by current-year fiscal appropriation.

**Cause.** The carryover funds are project budgets approved in earlier years but not fully executed, mainly for equipment purchase and renewal and infrastructure maintenance.

**Impact.** This expenditure depends on non-recurring funds and is unsustainable. If no new carryover or special appropriation is added, similar projects will face funding gaps.

#### HyDE-RAG

**Observation.** In Department D02's 2024 project expenditure, 252.50 comes from prior-year carryover funds and is not covered by current-year fiscal appropriation.

**Cause.** The 252.50 is used as unit funds for equipment purchase and renewal (181.69) and infrastructure maintenance (70.81). No evidence shows whether it comes from one-off items or recurring sources.

**Impact.** Carryover funds account for about 2.20% of total expenditure and 3.60% of project expenditure. This is a non-recurring source. Without new carryover or extra appropriation, this spending may not be sustainable.

#### CoT-RAG

**Observation.** Department D02's total 2024 expenditure is 11,460.62, of which 252.50 comes from prior-year carryover funds. This equals the difference between the fiscal appropriation summary and the revenue and expenditure summary.

**Observation.** The carryover funds are used for equipment purchase and renewal (181.69) and infrastructure maintenance (70.81).

**Observation.** In 2023 the department's prior-year carryover was 467.93, so carryover dropped by 215.43 in 2024.

**Cause.** The carryover comes from routine year-end balances and is consistent with normal carryover management. The specific formation items are not given in the evidence.

**Impact.** Carryover funds account for only 2.20% of total expenditure and 3.60% of project expenditure, and the share is falling year over year. They are not a main funding source and do not affect budget sustainability.

#### TRACE (Ours)

**Observation.** Of Department D02's 2024 total expenditure of 11,460.62, 252.50 comes from prior-year carryover funds rather than current-year fiscal appropriation, which matches the 252.50 gap between the fiscal appropriation summary (11,208.12) and the overall total.

**Cause.** The 252.50 is used entirely for project expenditure: 181.69 for equipment purchase and renewal and 70.81 for infrastructure maintenance. The materials do not explain how these carryover funds were formed.

**Impact.** This part of project expenditure depends on carryover funds, which may be non-recurring. If no new carryover or special arrangement is available in later years, these projects may not be continuously funded, creating a potential funding gap.

### Analysis

This case tests latent risk discovery. The key point is not just restating numbers but recognising that 252.50 of project expenditure is funded by prior-year carryover rather than current-year appropriation.

TRACE identifies this risk anchor and links it to a potential funding gap, while keeping the formation reason as "not explained".

CoT-RAG finds the carryover but weakens the risk. It brings in the 2023 carryover (467.93), which the question did not ask for, and concludes that a small, falling share "does not affect budget sustainability". From an audit perspective, funding projects with possibly non-recurring carryover should still raise a sustainability concern.

Closed-book Generation, BM25-RAG, and Naïve Dense RAG err the other way. They state that the funds are unspent 2023 balances, procurement delays, or unfinished approved projects. These explanations are plausible, but the evidence only shows that the funds come from carryover. It says nothing about why. HyDE-RAG keeps the evidence boundary on the cause but adds ratios that the reference does not need.

### One-sentence takeaway

TRACE correctly identifies that 252.50 of project expenditure depends on prior-year carryover rather than current-year appropriation and flags the potential funding gap, while other methods either downplay this risk or invent unsupported causes.

---

## Case 2: Suppressing Pseudo-Risk Generation under Incomplete Evidence

Demo row: `R000026-178f375d` (Department D05, FY2024). Gold evidence: revenue and expenditure summary (`…-0002`), project expenditure table (`…-0006`), and budget explanation (`…-0008`) of document `890c36fba0`.

### Query

Regarding how Department D05 reports the use of its 2024 special-account funds: the revenue and expenditure summary shows special-account revenue of 89.75, but the project expenditure table lists only 57.39 from the special account. Please explain where the difference went, which expenditure category it belongs to, and whether it is carried over.

### Reference Answer

**Observation.** The revenue and expenditure summary reports special-account revenue of 89.75.

**Observation.** The project expenditure table lists special-account funds of 57.39.

**Observation.** The difference of 32.36 is not listed in the project expenditure table. Correspondingly, project expenditure in the revenue and expenditure summary (8,231.23) exceeds the total of the project expenditure table (8,198.87) by the same amount.

**Cause.** The materials do not explain the use or carryover arrangement of the 32.36 of special-account funds.

**Impact.** Disclosure of the source and use of special-account funds is incomplete, which hinders review of whether these funds were used completely.

### Model Outputs (illustrative)

#### Closed-book Generation

**Observation.** The department received 89.75 of special-account funds but listed only 57.39 in project expenditure, leaving 32.36 unaccounted for.

**Cause.** The remaining 32.36 was probably used for basic expenditure such as office operations without its source being shown separately.

**Cause.** Alternatively, the funds may not have been executed and were carried over to 2025.

**Impact.** Using special-account funds outside the project budget may breach dedicated-use rules and creates compliance risk.

#### BM25-RAG

**Observation.** Special-account revenue is 89.75, while the project expenditure table lists 57.39 for training and meetings, a difference of 32.36.

**Cause.** The 32.36 was not spent in 2024 and was carried forward as a year-end balance.

**Impact.** The unexecuted special-account funds indicate low budget execution efficiency and idle funds.

#### Naïve Dense RAG

**Observation.** Special-account revenue is 91.14, while the project expenditure table lists 57.39, a difference of 33.75.

**Cause.** The difference arises because part of the special-account funds is reported under basic expenditure rather than project expenditure.

**Impact.** Inconsistent reporting of special-account funds makes it hard to assess budget execution.

#### HyDE-RAG

**Observation.** Special-account revenue is 89.75, of which 57.39 is listed in the project expenditure table, leaving 32.36.

**Cause.** As a culture, tourism, and sports department, part of the special-account funds likely comes from sports lottery public welfare funds and venue rental income. These are managed under separate rules and are therefore excluded from the project table.

**Impact.** Mixing lottery welfare funds with other special-account funds may weaken dedicated-use supervision.

#### CoT-RAG

**Observation.** Special-account revenue is 89.75 and the project expenditure table lists 57.39, a difference of 32.36, which is 36.06% of special-account revenue.

**Observation.** In 2023, special-account revenue of 91.14 was fully listed in the project expenditure table, so the gap is new in 2024.

**Cause.** Because unit funds of 354.81 were added to the information system project in 2024, the 32.36 was likely reallocated to that project and recorded as unit funds.

**Impact.** Moving special-account funds into other projects without disclosure suggests possible misappropriation and should be treated as a significant audit finding.

#### TRACE (Ours)

**Observation.** The revenue and expenditure summary reports special-account revenue of 89.75, while the project expenditure table lists 57.39, a difference of 32.36.

**Observation.** Project expenditure in the revenue and expenditure summary (8,231.23) exceeds the project expenditure table total (8,198.87) by the same 32.36.

**Cause.** The materials do not explain where the 32.36 went, which category it belongs to, or whether it is carried over.

**Impact.** The disclosure of special-account funds is incomplete between the two tables, which may hinder review of whether these funds were fully and properly used.

### Analysis

This case tests whether a model can suppress pseudo-risks when the evidence is incomplete. The reference answer confirms the 32.36 gap and its appearance in both tables, but it deliberately leaves the destination, category, and carryover status unresolved.

TRACE preserves this boundary. It verifies both appearances of the 32.36 gap through cross-table reconciliation, and it states that the materials do not explain the cause.

The baselines fill the gap with unsupported stories. Closed-book Generation and BM25-RAG assert that the funds were used for basic expenditure or carried over. Naïve Dense RAG retrieves the 2023 document of the same template and misreads special-account revenue as 91.14, producing a wrong gap of 33.75. HyDE-RAG invents lottery welfare funds and venue rental income that appear nowhere in the evidence. CoT-RAG is the most harmful. It correctly notes that the 2023 funds were fully listed, but it then links the gap to an unrelated unit-funds amount (354.81, which comes from prior-year carryover) and escalates the finding to possible misappropriation.

TRACE is better not because it says less, but because it says only what the evidence supports. It still reports the verifiable risk, an incomplete disclosure of special-account funds.

### One-sentence takeaway

TRACE confirms the 32.36 special-account discrepancy across both tables without guessing its cause, while other methods either misread numbers from a same-template document or turn an unexplained gap into fabricated causes or misappropriation claims.
