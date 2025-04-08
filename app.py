import streamlit as st
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your model and vectorizer
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page Title
st.title("United States Wikipedia Sentiment Analyzer")
st.subheader("📊 Powered by Random Forest + TextBlob")

# Description
st.markdown("""
Analyze the sentiment of Wikipedia-style text content related to the **United States**.  
This app uses a **Random Forest classifier** trained on U.S.-specific data and shows deep insight using **TextBlob metrics**.
""")

# Default Input
default_text = "The United States has a diverse culture and a strong global influence in many areas."
user_text = st.text_area("📝 Type or paste your sentence here:", default_text)

# Generate Word Cloud
if user_text:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_text)
    st.image(wordcloud.to_array(), caption="🌀 Word Cloud", use_container_width=True)

# Sentiment Analysis Button
if st.button("🔍 Analyze Sentiment"):
    # Vectorize Input
    input_vector = vectorizer.transform([user_text])
    prob = model.predict_proba(input_vector)[0]
    class_names = model.classes_  # ['Negative', 'Positive']
    pred_index = np.argmax(prob)
    sentiment = class_names[pred_index]
    sentiment_color = "green" if sentiment == "Positive" else "red"

    # Show Sentiment Result
    st.markdown(f"### ✅ Predicted Sentiment: :{sentiment_color}[{sentiment}]")

    # --- Bar Chart Visualization ---
    st.subheader("📈 Prediction Probability - Bar Chart")
    fig_bar, ax_bar = plt.subplots()
    sns.barplot(x=class_names, y=prob, palette='viridis', ax=ax_bar)
    ax_bar.set_ylabel("Probability")
    ax_bar.set_xlabel("Sentiment")
    ax_bar.set_ylim(0, 1)
    for i, v in enumerate(prob):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    st.pyplot(fig_bar)

    # --- Pie Chart Visualization ---
    st.subheader("🥧 Prediction Probability - Pie Chart")
    fig_pie, ax_pie = plt.subplots()
    pie_colors = ['lightcoral', 'lightgreen'] if 'Positive' in class_names else ['lightblue', 'lightpink']
    ax_pie.pie(prob, labels=class_names, autopct='%1.1f%%', colors=pie_colors, startangle=90)
    ax_pie.axis('equal')  # Equal aspect ratio ensures pie is circular
    st.pyplot(fig_pie)

# Footer
st.markdown("---")
st.markdown("📁 Model used: `rf_model.pkl` | ✒️ Vectorizer: `tfidf_vectorizer.pkl`")
st.markdown("👤 Built by Divya Sri for United States Wikipedia Sentiment Insights")
