# 🚗 AI Insurance Claim Processing System Using Multi-Agentic AI

A Streamlit application that uses a **6-agent LangGraph pipeline** to review an uploaded motor
insurance claim and produce a recommendation: **APPROVE**, **REJECT**, or **MANUAL REVIEW**.

> ⚠️ **This is a decision-support tool.** It gives an AI-generated *recommendation*, not a legally
> binding insurance decision. A human claims handler always makes the final call.

---

## 1. What This Project Does

Insurance companies receive a large volume of claims and employees have to manually verify
documents, policy details, claim information, fraud indicators, and supporting evidence — causing
slow processing, high manual workload, human error, and delays for genuine customers.

This project automates the **initial review** step:

1. User uploads claim documents (policy PDF, RC, driving licence, repair estimate, invoice,
   accident photos) and enters claim details through a Streamlit form.
2. Six agents process the claim in sequence.
3. The system displays each agent's findings and a final recommendation, with a downloadable
   report.

---

## 2. Project Structure

```
insurance-claim-agent/
├── app.py                                  # Streamlit app (UI + LangGraph pipeline)
├── requirements.txt
├── Insurance_Claim_Processing_Agent.ipynb  # Standalone notebook walkthrough
├── generate_samples.py                     # Generates the sample test PDFs below
├── sample_documents/                       # Ready-made PDFs for testing
│   ├── sample_policy_genuine.pdf
│   ├── sample_repair_estimate.pdf
│   ├── sample_invoice.pdf
│   ├── sample_rc.pdf
│   └── sample_policy_suspicious.pdf
├── .env.example
└── README.md
```

---

## 3. How to Run on Windows

### Step 1 — Create a project folder and put the files inside it
```bat
mkdir insurance-claim-agent
cd insurance-claim-agent
```
(Copy `app.py`, `requirements.txt`, the notebook, `generate_samples.py`, `sample_documents/`,
`.env.example`, and `README.md` into this folder.)

### Step 2 — Create a virtual environment
```bat
python -m venv venv
```

### Step 3 — Activate the virtual environment
```bat
venv\Scripts\activate
```

### Step 4 — Install requirements
```bat
pip install -r requirements.txt
```

### Step 5 — Create your .env file
```bat
copy .env.example .env
```

### Step 6 — Add your API key
Open `.env` in a text editor and replace the placeholder:
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### Step 7 — Run Streamlit
```bat
streamlit run app.py
```

### Step 8 — Open the application
Streamlit will print a local URL (usually `http://localhost:8501`) — open it in your browser.

> To regenerate the sample test PDFs yourself: `pip install reportlab` then
> `python generate_samples.py` (reportlab is only needed for this optional script, not for the
> app itself).

---

## 4. Sample Test Cases

### Test Case 1 — Normal Genuine Claim
- **Claimant Name:** Rahul Sharma
- **Policy Number:** POL-2026-88213
- **Vehicle Number:** DL8CAF1234
- **Claim Type:** Accident Damage
- **Claim Amount:** ₹23,000
- **Description:** "My car was hit from behind while parked outside my office on 12 Aug 2026. The
  rear bumper and tail lamp were damaged."
- **Documents to upload:** `sample_policy_genuine.pdf`, `sample_repair_estimate.pdf`,
  `sample_invoice.pdf`, `sample_rc.pdf`

**Expected output:** Policy Verification shows the claim matches the policy (comprehensive cover,
valid dates, matching vehicle number). Fraud/Risk shows **Low** risk (claim amount is reasonable,
multiple supporting documents provided). Final recommendation: **APPROVE**.

### Test Case 2 — Suspicious / Fraud-Risk Claim
- **Claimant Name:** Vikram Mehta
- **Policy Number:** POL-2024-11209
- **Vehicle Number:** HR26XY5678
- **Claim Type:** Accident Damage
- **Claim Amount:** ₹650,000
- **Description:** "My car's engine and interior were badly damaged in an accident. Requesting
  full repair cost."
- **Documents to upload:** `sample_policy_suspicious.pdf` only

**Expected output:** Policy Verification flags a concern — the policy is **Third-Party Only**
(no own-damage cover) and had already expired (validity 2024). Fraud/Risk shows rule-based flags
(`unusually_high_claim_amount`, `insufficient_supporting_documents`) and **High** risk. Final
recommendation: **REJECT** or **MANUAL REVIEW**.

### Test Case 3 — Missing / Invalid Documents
- **Claimant Name:** Anita Kumar
- **Policy Number:** POL-2026-55001
- **Vehicle Number:** KA05MZ9999
- **Claim Type:** Accident Damage
- **Claim Amount:** ₹5,000
- **Description:** "Car was scratched in a parking lot."
- **Documents to upload:** *(don't upload anything)*

**Expected output:** The app blocks submission with a validation error: "Please upload at least
one supporting document." If you bypass this by uploading an unrelated/empty file, the Claim
Intake Agent still records the issue and the Decision Agent recommends **MANUAL REVIEW** with an
explanation citing the missing information.

---

## 5. Interview Explanation

**1. What business problem does this project solve?**
It automates the slow, manual, error-prone first pass of reviewing motor insurance claims —
checking documents, verifying against the policy, spotting fraud risk, and assessing the claim
amount — so human adjusters can focus on the claims that genuinely need judgment.

**2. Why is this Agentic AI?**
Instead of one large prompt trying to do everything, the work is broken into specialized agents,
each with a narrow responsibility and its own inputs/outputs, orchestrated as a graph where the
output of one agent becomes the input of the next.

**3. Why are multiple agents used?**
Each stage of claim review is a distinct skill: reading documents, checking a policy, flagging
fraud patterns, estimating repair cost reasonableness, and making a final call. Separating them
makes each step easier to test, debug, and explain individually — and mirrors how a real claims
team is organized.

**4. What does each agent do?**
- **Claim Intake** — organizes and validates the submitted form/documents.
- **Document Processing** — extracts text from PDFs (pypdf) and reads images via the LLM's vision
  capability.
- **Policy Verification** — checks the claim against the extracted policy terms.
- **Fraud/Risk** — combines deterministic rule checks (e.g. amount outliers, missing documents)
  with an LLM-generated risk narrative.
- **Claim Assessment** — compares the claimed amount against the repair estimate/invoice.
- **Decision** — synthesizes everything into APPROVE / REJECT / MANUAL REVIEW with reasoning.

**5. How does the workflow work?**
It's a sequential LangGraph `StateGraph`: `START → Claim Intake → Document Processing → Policy
Verification → Fraud/Risk → Claim Assessment → Decision → END`. All agents read from and write to
one shared state object.

**6. How are uploaded documents processed?**
Streamlit's file uploader accepts PDFs and images. PDFs are parsed with `pypdf` to extract text.
Images are sent as base64 to the same LLM (GPT-4o-mini, which supports vision) to describe damage
or read visible text — no separate OCR library needed.

**7. How is fraud/risk assessed?**
A mix of **cheap, deterministic rule checks** (claim amount thresholds, document count, policy
mismatch flags) and an **LLM-generated risk narrative** that reasons over all the evidence
together — similar to how the reference project combined rule-based verification with an LLM risk
agent.

**8. How is the final recommendation generated?**
The Decision Agent receives the policy verification, fraud/risk assessment, and claim assessment
as structured JSON, and an LLM prompt instructs it to weigh these and return one of the three
recommendations plus a plain-language explanation.

**9. Why were these libraries selected?**
`langgraph` for agent orchestration, `langchain-openai`/`langchain-core` for LLM calls (same
provider and pattern as the reference project), `pypdf` because PDF reading is genuinely new for
this use case, `Pillow` to validate uploaded images before sending them to the LLM, and
`streamlit` for the required UI. No database, message queue, or vector store was added since none
was genuinely necessary for a first version.

**10. What can be improved in the future?**
Real OCR/Document-AI for scanned or handwritten documents, a policy database instead of re-reading
PDFs each time, a persistent database to store claim history and support audits, a queue/dashboard
for claims routed to Manual Review, and multi-language document support.

---

## 6. Resume Description

> Built an AI-powered Insurance Claim Processing System using a 6-agent LangGraph pipeline
> (Claim Intake, Document Processing, Policy Verification, Fraud/Risk, Claim Assessment, Decision)
> that reads uploaded PDFs and images, cross-checks claims against policy terms, and generates an
> APPROVE/REJECT/MANUAL REVIEW recommendation with reasoning — delivered as an interactive
> Streamlit app with downloadable reports.
