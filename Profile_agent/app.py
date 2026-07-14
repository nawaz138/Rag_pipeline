import streamlit as st

from pypdf import PdfReader

from resume_agent import analyze_resume
from matcher_agent import match_candidate



st.set_page_config(
    page_title="Resume AI Recruiter",
    page_icon="🤖",
    layout="wide"
)



st.title("🤖 Resume Screening AI Chatbot")

st.write(
"""
Upload resume and analyze candidate
against job requirements.
"""
)



# Upload Resume

uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)



resume_text=""



if uploaded_file:


    reader=PdfReader(
        uploaded_file
    )


    for page in reader.pages:

        resume_text += page.extract_text()



    st.success(
        "Resume uploaded successfully"
    )



    if st.button(
        "Analyze Resume"
    ):


        with st.spinner(
            "AI analyzing resume..."
        ):


            profile=analyze_resume(
                resume_text
            )


            st.session_state.profile=profile



        st.subheader(
            "Candidate Profile"
        )


        col1,col2=st.columns(2)


        with col1:

            st.write(
                "### Name"
            )

            st.write(
                profile.name
            )


            st.write(
                "### Email"
            )

            st.write(
                profile.email
            )



        with col2:

            st.write(
                "### Experience"
            )

            st.write(
                profile.experience
            )



        st.write(
            "### Skills"
        )


        st.write(
            profile.skills
        )



        st.write(
            "### Projects"
        )


        st.write(
            profile.projects
        )





# JD Matching

st.divider()


jd=st.text_area(
    "Paste Job Description"
)



if st.button(
    "Evaluate Candidate"
):


    if "profile" not in st.session_state:

        st.warning(
            "Please analyze resume first"
        )


    else:


        result=match_candidate(
            st.session_state.profile,
            jd
        )


        st.subheader(
            "Recruiter Decision"
        )


        if result.Result=="Selected":

            st.success(
                result.Result
            )

        else:

            st.error(
                result.Result
            )


        st.write(
            result.Reason
        )
