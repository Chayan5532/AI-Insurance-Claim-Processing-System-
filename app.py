"""
AI Insurance Claim Processing System Using Multi-Agentic AI
=============================================================

A Streamlit app that runs an uploaded motor-insurance claim through a
6-agent LangGraph pipeline and produces a recommendation:
APPROVE / REJECT / MANUAL REVIEW.

This is a DECISION-SUPPORT tool. It does NOT make a legally binding
insurance decision -- a human claims handler always makes the final call.

Architecture (same LangGraph StateGraph style as the reference project):

    Claim Intake -> Document Processing -> Policy Verification
    -> Fraud/Risk Analysis -> Claim Assessment -> Decision
"""

import os
import json
import base64
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image
import io

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# --------------------------------------------------------------------------
# 0. Setup
# --------------------------------------------------------------------------

load_dotenv()

st.set_page_config(page_title="AI Insurance Claim Processing System", layout="wide")

# Supported file types for document upload
PDF_TYPES = ["pdf"]
IMAGE_TYPES = ["png", "jpg", "jpeg"]


def get_llm():
    """Create the LLM client. Raises a friendly error if the API key is missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Please add it to your .env file "
            "(see .env.example)."
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


# --------------------------------------------------------------------------
# 1. Shared Agent State
# --------------------------------------------------------------------------

class ClaimState(TypedDict):
    # Raw inputs
    claimant_info: dict
    policy_info: dict
    claim_description: str
    claim_amount: float
    claim_type: str
    documents: List[Dict[str, Any]]      # [{"name":..., "type": "pdf"/"image", "bytes": ...}]

    # Agent outputs
    intake_summary: Dict[str, Any]
    extracted_documents: Dict[str, Any]
    policy_verification: Dict[str, Any]
    fraud_risk: Dict[str, Any]
    claim_assessment: Dict[str, Any]
    decision: Dict[str, Any]

    # Housekeeping
    errors: List[str]


# --------------------------------------------------------------------------
# 2. Document reading helpers (stand-in for a real Document Processing
#    MCP tool -- here we use pypdf for PDFs and the LLM's vision ability
#    for images, reusing the SAME langchain_openai client as the rest of
#    the pipeline).
# --------------------------------------------------------------------------

def read_pdf_text(file_bytes: bytes) -> str:
    """Extract plain text from a PDF using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        return f"(could not read PDF: {e})"


def validate_image(file_bytes: bytes) -> bool:
    """Confirm the uploaded bytes are a real, openable image before sending to the LLM."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except Exception:
        return False


def describe_image_with_llm(llm, file_bytes: bytes, filename: str) -> str:
    """Ask the LLM (vision-capable gpt-4o-mini) to describe/extract info from an image."""
    try:
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        message = HumanMessage(content=[
            {"type": "text", "text": (
                f"This image is a document uploaded as part of a motor insurance claim "
                f"(filename: {filename}). If it is an accident photo, briefly describe the "
                f"visible vehicle damage. If it is a scanned document (like a driving licence "
                f"or RC), extract the key readable text fields. Keep the answer under 100 words."
            )},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ])
        resp = llm.invoke([message])
        return resp.content.strip()
    except Exception as e:
        return f"(could not read image: {e})"


# --------------------------------------------------------------------------
# 3. Agent 1: Claim Intake Agent
#    Organizes the raw form inputs into a clean structured summary.
#    Pure Python -- no LLM call needed, mirrors the simple state-setup
#    nodes in the reference project.
# --------------------------------------------------------------------------

def claim_intake_agent(state: ClaimState) -> dict:
    errors = []

    if not state.get("claim_description", "").strip():
        errors.append("Claim description is empty.")
    if not state.get("documents"):
        errors.append("No supporting documents were uploaded.")
    if state.get("claim_amount", 0) <= 0:
        errors.append("Claim amount must be greater than zero.")

    summary = {
        "claimant_name": state["claimant_info"].get("name", "Unknown"),
        "policy_number": state["policy_info"].get("policy_number", "Unknown"),
        "claim_type": state.get("claim_type", "Motor Insurance"),
        "claim_amount": state.get("claim_amount", 0),
        "num_documents": len(state.get("documents", [])),
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    return {"intake_summary": summary, "errors": errors}


# --------------------------------------------------------------------------
# 4. Agent 2: Document Processing Agent
#    Reads every uploaded document and extracts key information.
# --------------------------------------------------------------------------

def document_processing_agent(state: ClaimState) -> dict:
    if state.get("errors"):
        return {}  # skip if intake already found blocking errors

    llm = get_llm()
    extracted = {}

    for doc in state["documents"]:
        name = doc["name"]
        if doc["type"] == "pdf":
            raw_text = read_pdf_text(doc["bytes"])
            extracted[name] = {"kind": "pdf", "content": raw_text[:3000]}
        else:  # image
            if not validate_image(doc["bytes"]):
                extracted[name] = {"kind": "image", "content": "(file could not be opened as a valid image)"}
                continue
            description = describe_image_with_llm(llm, doc["bytes"], name)
            extracted[name] = {"kind": "image", "content": description}

    return {"extracted_documents": extracted}


# --------------------------------------------------------------------------
# 5. Agent 3: Policy Verification Agent
#    Checks whether the claim looks consistent with the policy document.
# --------------------------------------------------------------------------

POLICY_SYSTEM_PROMPT = """You are a Policy Verification agent for a motor insurance company.
Given the policy document text and the claim details, check whether the claim appears
consistent with the policy (coverage type, validity dates, insured vehicle, exclusions).

Return ONLY a JSON object (no markdown fences, no extra text) with keys:
  "policy_matches_claim" (true/false/"unclear"),
  "coverage_notes" (short string),
  "concerns" (list of short strings, empty list if none)
"""


def policy_verification_agent(state: ClaimState) -> dict:
    if state.get("errors"):
        return {}

    llm = get_llm()

    policy_text = ""
    for name, doc in state.get("extracted_documents", {}).items():
        if "policy" in name.lower():
            policy_text += doc["content"] + "\n"

    if not policy_text.strip():
        return {"policy_verification": {
            "policy_matches_claim": "unclear",
            "coverage_notes": "No policy document was uploaded or text could not be read.",
            "concerns": ["Policy document missing or unreadable."],
        }}

    resp = llm.invoke([
        SystemMessage(content=POLICY_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Policy document text:\n{policy_text[:3000]}\n\n"
            f"Claim details:\n{json.dumps(state['intake_summary'])}\n"
            f"Claim description: {state['claim_description']}"
        )),
    ])
    raw = resp.content.strip().strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"policy_matches_claim": "unclear", "coverage_notes": raw[:300], "concerns": []}

    return {"policy_verification": result}


# --------------------------------------------------------------------------
# 6. Agent 4: Fraud/Risk Agent
#    Rule-based checks (like the reference's Verification Agent) combined
#    with an LLM risk narrative.
# --------------------------------------------------------------------------

FRAUD_SYSTEM_PROMPT = """You are a Fraud/Risk assessment agent for a motor insurance company.
Given the claim details, extracted documents, policy verification result, and a list of
rule-based flags already detected, write a short risk assessment.

Return ONLY a JSON object (no markdown fences, no extra text) with keys:
  "risk_level" (one of "Low","Medium","High"),
  "risk_score" (integer 0-100),
  "reasoning" (short string, 2-3 sentences)
"""


def fraud_risk_agent(state: ClaimState) -> dict:
    if state.get("errors"):
        return {}

    # Rule-based flags (deterministic, same style as reference's verification_agent)
    flags = []
    claim_amount = state.get("claim_amount", 0)

    if claim_amount > 500000:
        flags.append("unusually_high_claim_amount")

    policy_ver = state.get("policy_verification", {})
    if policy_ver.get("policy_matches_claim") is False:
        flags.append("policy_mismatch")
    if policy_ver.get("concerns"):
        flags.append("policy_concerns_present")

    num_docs = state["intake_summary"].get("num_documents", 0)
    if num_docs < 2:
        flags.append("insufficient_supporting_documents")

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=FRAUD_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Claim details: {json.dumps(state['intake_summary'])}\n"
            f"Claim description: {state['claim_description']}\n"
            f"Policy verification: {json.dumps(policy_ver)}\n"
            f"Rule-based flags already detected: {flags}"
        )),
    ])
    raw = resp.content.strip().strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"risk_level": "Medium", "risk_score": 50, "reasoning": raw[:300]}

    result["rule_based_flags"] = flags
    return {"fraud_risk": result}


# --------------------------------------------------------------------------
# 7. Agent 5: Claim Assessment Agent
#    Summarizes the estimated claim impact using the repair estimate /
#    invoice against the claimed amount.
# --------------------------------------------------------------------------

ASSESSMENT_SYSTEM_PROMPT = """You are a Claim Assessment agent for a motor insurance company.
Given the claim amount, claim description, and any repair estimate/invoice document text,
summarize the estimated claim impact.

Return ONLY a JSON object (no markdown fences, no extra text) with keys:
  "estimated_amount_reasonable" (true/false/"unclear"),
  "summary" (short string, 2-3 sentences)
"""


def claim_assessment_agent(state: ClaimState) -> dict:
    if state.get("errors"):
        return {}

    estimate_text = ""
    for name, doc in state.get("extracted_documents", {}).items():
        lname = name.lower()
        if "estimate" in lname or "invoice" in lname or "bill" in lname:
            estimate_text += f"[{name}]\n{doc['content']}\n"

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=ASSESSMENT_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Claimed amount: {state.get('claim_amount')}\n"
            f"Claim description: {state['claim_description']}\n"
            f"Repair estimate / invoice text (if any):\n{estimate_text[:3000] or 'None provided'}"
        )),
    ])
    raw = resp.content.strip().strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"estimated_amount_reasonable": "unclear", "summary": raw[:300]}

    return {"claim_assessment": result}


# --------------------------------------------------------------------------
# 8. Agent 6: Decision Agent
#    Combines everything into a final APPROVE / REJECT / MANUAL REVIEW
#    recommendation with reasoning.
# --------------------------------------------------------------------------

DECISION_SYSTEM_PROMPT = """You are the final Decision agent of an AI-assisted insurance claim
processing system. You give a RECOMMENDATION only -- a human claims handler makes the final,
legally binding decision.

Based on the policy verification, fraud/risk assessment, and claim assessment provided,
recommend one of: "APPROVE", "REJECT", "MANUAL REVIEW".

Guidance:
- If risk is High, or policy clearly does not match, or documents are missing -> MANUAL REVIEW or REJECT
- If everything looks consistent and risk is Low -> APPROVE
- When uncertain, prefer MANUAL REVIEW over guessing

Return ONLY a JSON object (no markdown fences, no extra text) with keys:
  "recommendation" (one of "APPROVE","REJECT","MANUAL REVIEW"),
  "explanation" (string, 3-5 sentences explaining the reasoning in plain language)
"""


def decision_agent(state: ClaimState) -> dict:
    if state.get("errors"):
        return {"decision": {
            "recommendation": "MANUAL REVIEW",
            "explanation": "Claim could not be fully processed due to input errors: "
                            + "; ".join(state["errors"]),
        }}

    llm = get_llm()
    resp = llm.invoke([
        SystemMessage(content=DECISION_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Claim summary: {json.dumps(state['intake_summary'])}\n"
            f"Policy verification: {json.dumps(state.get('policy_verification', {}))}\n"
            f"Fraud/Risk assessment: {json.dumps(state.get('fraud_risk', {}))}\n"
            f"Claim assessment: {json.dumps(state.get('claim_assessment', {}))}"
        )),
    ])
    raw = resp.content.strip().strip("`").replace("json\n", "", 1)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"recommendation": "MANUAL REVIEW", "explanation": raw[:500]}

    return {"decision": result}


# --------------------------------------------------------------------------
# 9. Build & Compile the Graph
# --------------------------------------------------------------------------

def build_claim_graph():
    graph = StateGraph(ClaimState)

    graph.add_node("claim_intake", claim_intake_agent)
    graph.add_node("document_processing", document_processing_agent)
    graph.add_node("policy_verification", policy_verification_agent)
    graph.add_node("fraud_risk", fraud_risk_agent)
    graph.add_node("claim_assessment", claim_assessment_agent)
    graph.add_node("decision", decision_agent)

    graph.add_edge(START, "claim_intake")
    graph.add_edge("claim_intake", "document_processing")
    graph.add_edge("document_processing", "policy_verification")
    graph.add_edge("policy_verification", "fraud_risk")
    graph.add_edge("fraud_risk", "claim_assessment")
    graph.add_edge("claim_assessment", "decision")
    graph.add_edge("decision", END)

    return graph.compile()


# --------------------------------------------------------------------------
# 10. Streamlit UI
# --------------------------------------------------------------------------

def classify_uploaded_file(uploaded_file):
    """Return 'pdf' or 'image' based on file extension."""
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext in PDF_TYPES:
        return "pdf"
    elif ext in IMAGE_TYPES:
        return "image"
    return None


def build_report_text(state_result: dict) -> str:
    """Build a simple downloadable text/markdown report."""
    lines = []
    lines.append("# AI Insurance Claim Processing Report\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Claim Summary")
    for k, v in state_result.get("intake_summary", {}).items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Policy Verification")
    lines.append(f"```\n{json.dumps(state_result.get('policy_verification', {}), indent=2)}\n```")

    lines.append("\n## Fraud / Risk Assessment")
    lines.append(f"```\n{json.dumps(state_result.get('fraud_risk', {}), indent=2)}\n```")

    lines.append("\n## Claim Assessment")
    lines.append(f"```\n{json.dumps(state_result.get('claim_assessment', {}), indent=2)}\n```")

    decision = state_result.get("decision", {})
    lines.append("\n## Final Recommendation")
    lines.append(f"**{decision.get('recommendation', 'N/A')}**\n")
    lines.append(decision.get("explanation", ""))

    lines.append(
        "\n---\n*This is an AI-generated recommendation, not a legally binding "
        "insurance decision. A human claims handler must review and approve it.*"
    )
    return "\n".join(lines)


def main():
    st.title("🚗 AI Insurance Claim Processing System")
    st.caption("Multi-Agentic AI decision-support tool for motor insurance claims. "
               "This provides a RECOMMENDATION, not a final decision.")

    # ---------------- Sidebar: Claim Information ----------------
    with st.sidebar:
        st.header("📋 Claim Information")

        st.subheader("Claimant")
        claimant_name = st.text_input("Claimant Name")
        claimant_contact = st.text_input("Contact Number / Email")

        st.subheader("Policy")
        policy_number = st.text_input("Policy Number")
        vehicle_number = st.text_input("Vehicle Registration Number")

        st.subheader("Claim")
        claim_type = st.selectbox("Claim Type", ["Accident Damage", "Theft", "Third-Party", "Other"])
        claim_amount = st.number_input("Claim Amount (₹)", min_value=0.0, step=1000.0)

    # ---------------- Main Area ----------------
    st.subheader("1️⃣ Document Upload")
    st.write("Upload the insurance policy PDF and any supporting documents "
             "(RC, driving licence, repair estimate, invoice, accident photos).")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=PDF_TYPES + IMAGE_TYPES,
        accept_multiple_files=True,
    )

    st.subheader("2️⃣ Claim Details")
    claim_description = st.text_area(
        "Describe what happened",
        placeholder="E.g. Vehicle was hit from behind while parked on 12 Aug 2026...",
        height=120,
    )

    st.subheader("3️⃣ Process Claim")
    process_clicked = st.button("🚀 Process Claim", type="primary")

    if process_clicked:
        # --- Basic validation ---
        validation_errors = []
        if not claimant_name.strip():
            validation_errors.append("Claimant name is required.")
        if not policy_number.strip():
            validation_errors.append("Policy number is required.")
        if not claim_description.strip():
            validation_errors.append("Claim description is required.")
        if claim_amount <= 0:
            validation_errors.append("Claim amount must be greater than zero.")
        if not uploaded_files:
            validation_errors.append("Please upload at least one supporting document.")
        if not os.getenv("OPENAI_API_KEY"):
            validation_errors.append(
                "OPENAI_API_KEY is not set. Please add it to your .env file."
            )

        if validation_errors:
            for err in validation_errors:
                st.error(err)
            return

        # --- Prepare documents for the graph ---
        documents = []
        for f in uploaded_files:
            file_type = classify_uploaded_file(f)
            if file_type is None:
                st.warning(f"Skipping unsupported file type: {f.name}")
                continue
            documents.append({"name": f.name, "type": file_type, "bytes": f.getvalue()})

        initial_state = {
            "claimant_info": {"name": claimant_name, "contact": claimant_contact},
            "policy_info": {"policy_number": policy_number, "vehicle_number": vehicle_number},
            "claim_description": claim_description,
            "claim_amount": claim_amount,
            "claim_type": claim_type,
            "documents": documents,
            "errors": [],
        }

        st.subheader("4️⃣ Agent Processing Status")

        try:
            app_graph = build_claim_graph()
            result = None

            with st.status("Running multi-agent claim pipeline...", expanded=True) as status:
                st.write("🟢 Claim Intake Agent — organizing claim details...")
                st.write("🟢 Document Processing Agent — reading uploaded documents...")
                st.write("🟢 Policy Verification Agent — checking policy consistency...")
                st.write("🟢 Fraud/Risk Agent — assessing risk...")
                st.write("🟢 Claim Assessment Agent — evaluating claim amount...")
                st.write("🟢 Decision Agent — generating recommendation...")

                result = app_graph.invoke(initial_state)
                status.update(label="✅ Claim processing complete", state="complete")

        except Exception as e:
            st.error(f"Something went wrong while processing the claim: {e}")
            return

        if result.get("errors"):
            st.warning("⚠️ The following issues were found: " + "; ".join(result["errors"]))

        # ---------------- Results ----------------
        st.subheader("5️⃣ Extracted Information")
        with st.expander("View extracted document content", expanded=False):
            uploaded_bytes_by_name = {d["name"]: d["bytes"] for d in documents}
            for name, doc in result.get("extracted_documents", {}).items():
                st.markdown(f"**{name}** ({doc['kind']})")
                if doc["kind"] == "image" and name in uploaded_bytes_by_name:
                    try:
                        img = Image.open(io.BytesIO(uploaded_bytes_by_name[name]))
                        st.image(img, width=250)
                    except Exception:
                        pass
                st.text(doc["content"][:1000])

        st.subheader("6️⃣ Policy Verification")
        st.json(result.get("policy_verification", {}))

        st.subheader("7️⃣ Fraud / Risk Analysis")
        risk = result.get("fraud_risk", {})
        st.json(risk)
        if risk.get("risk_level"):
            color = {"Low": "green", "Medium": "orange", "High": "red"}.get(risk["risk_level"], "gray")
            st.markdown(f"**Risk Level:** :{color}[{risk['risk_level']}]")

        st.subheader("8️⃣ Claim Assessment")
        st.json(result.get("claim_assessment", {}))

        st.subheader("9️⃣ Final Decision")
        decision = result.get("decision", {})
        rec = decision.get("recommendation", "N/A")
        rec_color = {"APPROVE": "green", "REJECT": "red", "MANUAL REVIEW": "orange"}.get(rec, "gray")
        st.markdown(f"## :{rec_color}[{rec}]")
        st.write(decision.get("explanation", ""))
        st.caption("⚠️ This is an AI-generated recommendation, not a legally binding decision. "
                   "A human claims handler must review and approve it.")

        st.subheader("🔟 Download Report")
        report_text = build_report_text(result)
        st.download_button(
            label="📄 Download Claim Report (Markdown)",
            data=report_text,
            file_name=f"claim_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
