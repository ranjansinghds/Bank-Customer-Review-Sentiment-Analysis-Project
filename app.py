import streamlit as st
import pickle
import re
import string


# Load Model

with open(
    "bank_sentiment_model.pkl",
    "rb"
) as file:

    model = pickle.load(file)


# Load TF-IDF Vectorizer

with open(
    "tfidf_vectorizer.pkl",
    "rb"
) as file:

    vectorizer = pickle.load(file)


# Text Cleaning Function

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# Streamlit UI

st.set_page_config(
    page_title="Bank Customer Review Sentiment",
    page_icon="🏦",
    layout="centered"
)


st.title(
    "🏦 Bank Customer Review Sentiment Analysis"
)

st.write(
    "Enter a bank customer review and "
    "the NLP model will predict its sentiment."
)


review = st.text_area(
    "Enter Customer Review",
    height=180,
    placeholder="Example: The customer service was excellent..."
)


if st.button("Analyze Sentiment"):

    if review.strip() == "":

        st.warning(
            "Please enter a customer review."
        )

    else:

        cleaned_review = clean_text(review)

        review_vector = vectorizer.transform(
            [cleaned_review]
        )

        prediction = model.predict(
            review_vector
        )[0]

        probability = model.predict_proba(
            review_vector
        )[0]

        positive_probability = probability[1]

        if prediction == 1:

            st.success(
                "😊 Positive Review"
            )

            st.write(
                f"Positive Probability: "
                f"{positive_probability:.2%}"
            )

        else:

            st.error(
                "😞 Negative Review"
            )

            st.write(
                f"Negative Probability: "
                f"{probability[0]:.2%}"
            )