import streamlit as st
import whisper
import tempfile
import os
from openai import OpenAI

st.set_page_config(layout="wide")

client = OpenAI()

# SESSION STATE
for key in ["transcript", "crm_notes", "email", "next_steps", "deal_data"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# SIDEBAR
with st.sidebar:
    st.title("Deal Workspace")
    st.markdown("### Current Call")
    st.markdown("### Coming Soon")
    st.markdown("• Deal Summary")
    st.markdown("• Deal Risks")
    st.markdown("• Next Actions")

# MODEL
@st.cache_resource
def load_model():
    return whisper.load_model("base")

# LAYOUT
left, right = st.columns([1, 2])

# LEFT COLUMN
with left:
    st.header("Upload Audio")

    uploaded_file = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a"]
    )

    if uploaded_file:
        st.audio(uploaded_file)

    st.markdown("### Actions")

    transcribe_btn = st.button("Transcribe")
    crm_btn = st.button("CRM Notes")
    email_btn = st.button("Follow-Up Email")
    next_btn = st.button("Next Steps")
    deal_btn = st.button("Key Deal Data")

    status_box = st.empty()

# RIGHT COLUMN
with right:
    st.header("Transcript")
    st.text_area(
        "Transcript Output",
        value=st.session_state.transcript,
        height=200,
        label_visibility="collapsed"
    )
    st.button("Copy Transcript")

    st.markdown("---")

    st.header("CRM Notes")
    st.text_area(
        "CRM Notes Output",
        value=st.session_state.crm_notes,
        height=120,
        label_visibility="collapsed"
    )
    st.button("Copy CRM Notes")

    st.markdown("---")

    st.header("Follow-Up Email")
    st.text_area(
        "Follow-Up Email Output",
        value=st.session_state.email,
        height=160,
        label_visibility="collapsed"
    )
    st.button("Copy Email")

    st.markdown("---")

    st.header("Next Steps")
    st.text_area(
        "Next Steps Output",
        value=st.session_state.next_steps,
        height=120,
        label_visibility="collapsed"
    )
    st.button("Copy Next Steps")

    st.markdown("---")

    st.header("Key Deal Data")
    st.text_area(
        "Key Deal Data Output",
        value=st.session_state.deal_data,
        height=120,
        label_visibility="collapsed"
    )
    st.button("Copy Deal Data")

# TRANSCRIBE
if transcribe_btn and uploaded_file:
    with status_box.container():
        with st.spinner("Transcribing..."):
            file_bytes = uploaded_file.getvalue()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(file_bytes)
                path = tmp.name

            model = load_model()
            result = model.transcribe(path, fp16=False, language="en")
            st.session_state.transcript = result["text"]

            os.remove(path)

    st.rerun()

# CRM NOTES
if crm_btn:
    if not st.session_state.transcript:
        with status_box.container():
            st.warning("Please generate a transcript first.")
    else:
        with status_box.container():
            with st.spinner("Generating CRM notes..."):
                try:
                    prompt = f"""
Create concise CRM notes (3–5 sentences) from this sales call transcript.

Include:
- company background if clear
- current solution
- key pain points
- important context
- next step agreed

Do not invent facts.

Transcript:
{st.session_state.transcript}
"""
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                    notes_text = response.output_text.strip()

                    if not notes_text:
                        st.error("CRM notes came back empty.")
                    else:
                        st.session_state.crm_notes = notes_text
                        st.rerun()

                except Exception as e:
                    st.error(f"CRM Notes error: {e}")

# FOLLOW UP EMAIL
if email_btn:
    if not st.session_state.transcript:
        with status_box.container():
            st.warning("Please generate a transcript first.")
    else:
        with status_box.container():
            with st.spinner("Generating follow-up email..."):
                try:
                    prompt = f"""
Write a short, natural follow-up email after this sales call.

Requirements:
- conversational
- short
- no AI-sounding language
- minimal formatting
- sound like a real sales rep wrote it

Transcript:
{st.session_state.transcript}
"""
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                    email_text = response.output_text.strip()

                    if not email_text:
                        st.error("Follow-up email came back empty.")
                    else:
                        st.session_state.email = email_text
                        st.rerun()

                except Exception as e:
                    st.error(f"Follow-Up Email error: {e}")

# NEXT STEPS
if next_btn:
    if not st.session_state.transcript:
        with status_box.container():
            st.warning("Please generate a transcript first.")
    else:
        with status_box.container():
            with st.spinner("Generating next steps..."):
                try:
                    prompt = f"""
From this sales call transcript, extract the clear next steps.

Return a short bullet list including:
- follow-ups
- actions promised
- scheduling items
- decisions pending

Transcript:
{st.session_state.transcript}
"""
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                    next_text = response.output_text.strip()

                    if not next_text:
                        st.error("Next steps came back empty.")
                    else:
                        st.session_state.next_steps = next_text
                        st.rerun()

                except Exception as e:
                    st.error(f"Next Steps error: {e}")

# DEAL DATA
if deal_btn:
    if not st.session_state.transcript:
        with status_box.container():
            st.warning("Please generate a transcript first.")
    else:
        with status_box.container():
            with st.spinner("Extracting key deal data..."):
                try:
                    prompt = f"""
Extract key deal data from this sales call transcript.

Return concise structured info for:
- Company
- Contact Name
- Decision Maker
- Current Provider / Current Solution
- Pain Points
- Products Discussed
- Timeline
- Next Step

If something is not stated clearly, say "Not stated".

Transcript:
{st.session_state.transcript}
"""
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=prompt
                    )

                    deal_text = response.output_text.strip()

                    if not deal_text:
                        st.error("Key deal data came back empty.")
                    else:
                        st.session_state.deal_data = deal_text
                        st.rerun()

                except Exception as e:
                    st.error(f"Key Deal Data error: {e}")