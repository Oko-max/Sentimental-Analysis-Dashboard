import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
from collections import Counter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

# --- Configuration ---
# Set page config for a wider layout and title
st.set_page_config(layout="wide", page_title=" Sentiment Analysis Dashboard")

# --- Load Sentiment Analysis Model ---
@st.cache_resource
def load_sentiment_model():
    """
    Loads the pre-trained sentiment analysis model from Hugging Face.
    Uses 'cardiffnlp/twitter-roberta-base-sentiment-latest' for multi-class classification.
    The model is cached to avoid reloading on every rerun.
    """
    try:
        # Using a model known for multi-class sentiment (positive, negative, neutral)
        # This model outputs labels like 'LABEL_0', 'LABEL_1', 'LABEL_2' which map to negative, neutral, positive.
        # We will map these labels for better readability.
        sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}. Please check your internet connection and ensure 'torch' or 'tensorflow' is installed.")
        st.stop()

sentiment_analyzer = load_sentiment_model()

# Map model output labels to human-readable labels
# Based on the model card for 'cardiffnlp/twitter-roberta-base-sentiment-latest':
# LABEL_0: Negative
# LABEL_1: Neutral
# LABEL_2: Positive
LABEL_MAPPING = {
    'LABEL_0': 'Negative',
    'LABEL_1': 'Neutral',
    'LABEL_2': 'Positive',
    'neutral': 'Neutral',  # Added for robustness if model directly outputs 'neutral'
    'positive': 'Positive', # Added for robustness
    'negative': 'Negative'  # Added for robustness
}

# --- Keyword Extraction (Basic) ---
def extract_keywords(text, num_keywords=5):
    """
    A basic keyword extraction function.
    It tokenizes the text, removes common stopwords, and returns the most frequent words.
    For more advanced keyword extraction, consider using libraries like Rake-NLTK, spaCy, or YAKE.
    """
    if not text.strip():
        return []

    # Simple tokenization and lowercasing
    words = re.findall(r'\b\w+\b', text.lower())

    # Basic English stopwords (can be expanded)
    stopwords = set([
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
        "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
        "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
        "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
        "while", "of", "at", "by", "for", "with", "about", "against", "between", "into",
        "through", "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "d",
        "ll", "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
        "weren", "won", "wouldn"
    ])

    # Filter out stopwords and non-alphabetic words
    filtered_words = [word for word in words if word.isalpha() and word not in stopwords]

    # Count word frequencies
    word_counts = Counter(filtered_words)

    # Return the most common words
    return [word for word, count in word_counts.most_common(num_keywords)]

# --- Sentiment Analysis Function ---
def analyze_text_sentiment(text):
    """
    Analyzes the sentiment of a single piece of text.
    Returns the predicted label (e.g., 'Positive', 'Negative', 'Neutral')
    and the confidence score, along with a more detailed explanation including keywords.
    """
    if not text.strip():
        return "N/A", 0.0, "No text provided for analysis."

    try:
        raw_result = sentiment_analyzer(text)
        if not raw_result:
            return "N/A", 0.0, "The sentiment model did not return a valid result for this text."

        result = raw_result[0]
        raw_label = result['label']
        score = result['score']
        # Map label for readability, handling both LABEL_X and direct string outputs
        label = LABEL_MAPPING.get(raw_label, raw_label)

        # Extract keywords for the current text
        keywords = extract_keywords(text, num_keywords=3) # Get top 3 keywords for explanation
        keywords_str = f" Keywords: {', '.join(keywords)}." if keywords else ""

        explanation = ""
        if label == 'Positive':
            explanation = f"The model identified strong positive indicators within the text, suggesting a favorable sentiment (Confidence: {score:.2f}).{keywords_str}"
        elif label == 'Negative':
            explanation = f"The model identified strong negative indicators within the text, suggesting an unfavorable sentiment (Confidence: {score:.2f}).{keywords_str}"
        elif label == 'Neutral':
            explanation = f"The text appears to be objective, factual, or lacks significant emotional vocabulary, leading to a neutral classification (Confidence: {score:.2f}).{keywords_str}"
        else:
            # This block should now be rarely hit if the model consistently outputs expected labels
            explanation = f"The model returned an unexpected label '{raw_label}'. Sentiment could not be clearly interpreted based on expected categories (Confidence: {score:.2f})."
            st.warning(f"Unexpected raw sentiment label encountered: {raw_label} for text: '{text[:50]}...'")

        return label, score, explanation
    except Exception as e:
        st.error(f"Error analyzing sentiment for text: '{text[:50]}...' - {e}")
        return "Error", 0.0, "An error occurred during analysis due to a processing issue."

# --- PDF Export Function ---
def create_pdf_report(df_results, sentiment_distribution_plot_buffer, sentiment_trend_plot_buffer=None, confidence_histogram_plot_buffer=None):
    """
    Generates a PDF report from the sentiment analysis results.
    Includes a title, a table of results, and the sentiment distribution plot.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Sentiment Analysis Report", styles['h1']))
    story.append(Spacer(1, 0.2 * inch))

    # Results Table
    if not df_results.empty:
        story.append(Paragraph("Analysis Results:", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))

        # Prepare table data, including the new 'Explanation' column
        data = [df_results.columns.tolist()] + df_results.values.tolist()

        # Adjust column widths to accommodate the new 'Explanation' column
        # Original: [2.5*inch, 1.5*inch, 1.5*inch] for Text, Sentiment, Confidence Score
        # New: Text, Sentiment, Confidence Score, Explanation
        # Distribute 8.5 inches (page width) - margins (1 inch each side) = 6.5 inches
        # Let's use relative widths or fixed for some and flexible for others
        # Text: 1.8, Sentiment: 0.9, Confidence: 0.9, Explanation: 2.9
        col_widths = [1.8*inch, 0.9*inch, 0.9*inch, 2.9*inch] # Total 6.5 inch

        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'), # Align text left for readability
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    # Sentiment Distribution Plot
    if sentiment_distribution_plot_buffer:
        story.append(Paragraph("Sentiment Distribution:", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))
        img = Image(sentiment_distribution_plot_buffer, width=6*inch, height=4*inch) # Explicitly set size
        story.append(img)
        story.append(Spacer(1, 0.2 * inch))

    # Sentiment Trend Plot
    if sentiment_trend_plot_buffer:
        story.append(Paragraph("Sentiment Score Trend:", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))
        img_trend = Image(sentiment_trend_plot_buffer, width=6*inch, height=4*inch) # Explicitly set size
        story.append(img_trend)
        story.append(Spacer(1, 0.2 * inch))

    # Confidence Histogram Plot
    if confidence_histogram_plot_buffer:
        story.append(Paragraph("Confidence Score Histogram:", styles['h2']))
        story.append(Spacer(1, 0.1 * inch))
        img_hist = Image(confidence_histogram_plot_buffer, width=6*inch, height=4*inch) # Explicitly set size
        story.append(img_hist)
        story.append(Spacer(1, 0.2 * inch))

    # Model Limitations
    story.append(Paragraph("Model Limitations and Confidence Thresholds:", styles['h2']))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        "This dashboard uses the 'cardiffnlp/twitter-roberta-base-sentiment-latest' model from Hugging Face. "
        "This model was fine-tuned on Twitter data, so its performance might vary on different types of text "
        "(e.g., formal documents, highly technical content). "
        "Confidence scores indicate the model's certainty; lower scores suggest less certainty. "
        "A general threshold for high confidence is often considered above 0.8, but this can vary by application.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2 * inch))

    try:
        doc.build(story)
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None # Return None if PDF generation fails
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.title(" Fire4s Sentiment Analysis Dashboard")
st.markdown(
    """
    Understand the emotional tone of your text data.
    Analyze customer reviews, social media posts, or any text content.
    """
)

# Sidebar for file upload
st.sidebar.header("Upload Text File (CSV/TXT)")
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV or TXT file for batch processing.",
    type=["csv", "txt"]
)
st.sidebar.markdown("---")
st.sidebar.info(
    "**Tip:** For CSV files, ensure the text column is clearly identifiable. "
    "For TXT files, each line will be treated as a separate text entry."
)

# Main content area for direct text input
st.header("Analyze Single Text or Batch Input")
input_text = st.text_area(
    "Enter multiple texts here for sentiment analysis (one per line):", # Updated prompt
    height=150,
    placeholder="e.g.,\nThis product is amazing! I love it.\nThe service was okay, nothing special.\nI am very disappointed with the quality."
)

if uploaded_file is not None:
    st.info(f"File '{uploaded_file.name}' uploaded. Processing will prioritize the file content.")
    if uploaded_file.type == "text/csv":
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.subheader("CSV File Preview:")
            st.dataframe(df_uploaded.head())

            text_column = st.selectbox(
                "Select the column containing text for analysis:",
                df_uploaded.columns
            )
            texts_to_analyze = df_uploaded[text_column].astype(str).tolist()
        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")
            texts_to_analyze = []
    elif uploaded_file.type == "text/plain":
        texts_to_analyze = uploaded_file.read().decode("utf-8").splitlines()
        texts_to_analyze = [line.strip() for line in texts_to_analyze if line.strip()] # Remove empty lines
        st.info(f"Loaded {len(texts_to_analyze)} lines from the TXT file.")
    else:
        st.error("Unsupported file type. Please upload a CSV or TXT file.")
        texts_to_analyze = []
else:
    # If no file uploaded, use the direct text input, splitting by lines
    if input_text.strip():
        texts_to_analyze = [line.strip() for line in input_text.splitlines() if line.strip()]
    else:
        texts_to_analyze = []

process_button = st.button("Analyze Sentiment")

if process_button and not texts_to_analyze:
    st.warning("Please enter some text or upload a file to analyze.")

if process_button and texts_to_analyze:
    st.subheader("Analysis Results:")
    results = []
    progress_bar = st.progress(0)
    for i, text in enumerate(texts_to_analyze):
        label, score, explanation = analyze_text_sentiment(text) # Get explanation
        results.append({
            "Text": text,
            "Sentiment": label,
            "Confidence Score": float(f"{score:.4f}"), # Convert to float for plotting
            "Explanation": explanation # Add explanation to results
        })
        progress_bar.progress((i + 1) / len(texts_to_analyze))

    df_results = pd.DataFrame(results)

    # Function to apply background color based on sentiment
    def color_sentiment(val):
        if val == 'Positive':
            return 'background-color: #d4edda' # Light green
        elif val == 'Negative':
            return 'background-color: #f8d7da' # Light red
        elif val == 'Neutral':
            return 'background-color: #fff3cd' # Light yellow
        return '' # No color

    # Apply styling to the DataFrame
    st.dataframe(df_results.style.applymap(color_sentiment, subset=['Sentiment']), use_container_width=True)

    # --- Visualization: Sentiment Distribution ---
    st.subheader("Sentiment Distribution")
    sentiment_distribution_plot_buffer = None
    if not df_results.empty:
        sentiment_counts = df_results['Sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
        fig_dist, ax_dist = plt.subplots(figsize=(8, 5))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax_dist)
        ax_dist.set_title("Overall Sentiment Distribution")
        ax_dist.set_xlabel("Sentiment")
        ax_dist.set_ylabel("Number of Texts")
        st.pyplot(fig_dist)

        # Save plot to buffer for PDF
        sentiment_distribution_plot_buffer = io.BytesIO()
        fig_dist.savefig(sentiment_distribution_plot_buffer, format='png', bbox_inches='tight')
        plt.close(fig_dist) # Close the plot to free memory
        sentiment_distribution_plot_buffer.seek(0)
    else:
        st.info("No sentiment data to visualize.")

    # --- New Visualization: Sentiment Score Trend (Linear Graph) ---
    st.subheader("Sentiment Score Trend")
    sentiment_trend_plot_buffer = None
    if len(df_results) > 1: # Only show trend if there are multiple texts
        fig_trend, ax_trend = plt.subplots(figsize=(10, 6))
        # Use index as x-axis for trend, and confidence score as y-axis
        ax_trend.plot(df_results.index, df_results['Confidence Score'], marker='o', linestyle='-', color='skyblue')
        ax_trend.set_title("Confidence Score Trend Across Texts")
        ax_trend.set_xlabel("Text Index")
        ax_trend.set_ylabel("Confidence Score")
        ax_trend.set_ylim(0, 1) # Confidence scores are between 0 and 1
        ax_trend.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig_trend)

        # Save plot to buffer for PDF
        sentiment_trend_plot_buffer = io.BytesIO()
        fig_trend.savefig(sentiment_trend_plot_buffer, format='png', bbox_inches='tight')
        plt.close(fig_trend) # Close the plot to free memory
        sentiment_trend_plot_buffer.seek(0)
    elif not df_results.empty:
        st.info("Enter multiple texts to see the sentiment score trend.")
    else:
        st.info("No sentiment data to visualize trend.")

    # --- New Visualization: Confidence Score Histogram ---
    st.subheader("Confidence Score Distribution (Histogram)")
    confidence_histogram_plot_buffer = None
    if not df_results.empty:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(df_results['Confidence Score'], bins=10, kde=True, ax=ax_hist, color='lightcoral')
        ax_hist.set_title("Distribution of Confidence Scores")
        ax_hist.set_xlabel("Confidence Score")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_xlim(0, 1) # Confidence scores are between 0 and 1
        st.pyplot(fig_hist)

        # Save plot to buffer for PDF
        confidence_histogram_plot_buffer = io.BytesIO()
        fig_hist.savefig(confidence_histogram_plot_buffer, format='png', bbox_inches='tight')
        plt.close(fig_hist) # Close the plot to free memory
        confidence_histogram_plot_buffer.seek(0)
    else:
        st.info("No confidence score data to visualize.")


    # --- Keyword Extraction & Explanation ---
    st.subheader("Keyword Insights & Explanation")
    if not df_results.empty:
        # Combine all analyzed text for overall keyword extraction
        all_text = " ".join(df_results['Text'].tolist())
        keywords = extract_keywords(all_text, num_keywords=10)
        if keywords:
            st.write(f"**Most Frequent Keywords Across All Texts:** {', '.join(keywords)}")
        else:
            st.info("No significant keywords found (or text too short/empty).")

        st.markdown(
            """
            **Understanding Confidence Scores:**
            The confidence score (ranging from 0 to 1) indicates how certain the model is about its sentiment prediction.
            A score closer to 1 means higher confidence. For example, a "Positive" sentiment with a score of 0.95
            is more certain than one with 0.60.
            """
        )
        st.markdown(
            """
            **Sentiment Rationale (Basic):**
            The model classifies text into 'Positive', 'Negative', or 'Neutral' based on patterns learned from a vast dataset.
            Words and phrases commonly associated with positive or negative emotions (e.g., "excellent", "terrible", "happy", "sad")
            strongly influence the classification. Neutral sentiment often indicates factual statements or a lack of strong emotional tone.
            """
        )
    else:
        st.info("No results to provide keyword insights or explanations.")


    # --- Export Results ---
    st.subheader("Export Results")
    if not df_results.empty:
        col1, col2, col3 = st.columns(3)

        # CSV Export
        csv_export = df_results.to_csv(index=False).encode('utf-8')
        col1.download_button(
            label="Download Results as CSV",
            data=csv_export,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

        # JSON Export
        json_export = df_results.to_json(orient="records", indent=4).encode('utf-8')
        col2.download_button(
            label="Download Results as JSON",
            data=json_export,
            file_name="sentiment_analysis_results.json",
            mime="application/json"
        )

        # PDF Export
        pdf_buffer = create_pdf_report(df_results, sentiment_distribution_plot_buffer, sentiment_trend_plot_buffer, confidence_histogram_plot_buffer)
        if pdf_buffer:
            col3.download_button(
                label="Download Report as PDF",
                data=pdf_buffer,
                file_name="sentiment_analysis_report.pdf",
                mime="application/pdf"
            )
        else:
            col3.info("PDF generation failed.")
    else:
        st.info("No results to export yet.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Hugging Face Transformers.")
