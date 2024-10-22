import re
import os
import matplotlib.pyplot as plt
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer
import nltk
import pyLDAvis
import pyLDAvis.gensim_models
from multiprocessing import cpu_count


# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to lemmatize text
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in text]

# Function to preprocess text for LDA
def preprocess_text(text):
    # Define additional stopwords
    additional_stopwords = set([
        'experience', 'skills', 'team', 'will',
        'ability', 'responsibilities', 'qualifications', 'requirements',
        'provide', 'knowledge', 'position', 'candidate', 'environment',
        'working', 'years', 'preferred', 'including', 'degree', 'related',
        'field', 'bachelor', 'minimum', 'strong', 'excellent', 'communication',
        'language', 'proficiency', 'job', 'role', 'must', 'plus', 'well',
        'good', 'verbal', 'written', 'one', 'new', 'also', 'make', 'may',
        'etc', 'many', 'time', 'using', 'us', 'within', 'across'
    ])
    stop_words = STOPWORDS.union(additional_stopwords)

    # Tokenize and clean-up text
    tokens = simple_preprocess(text, deacc=True)
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 3]
    # Lemmatize tokens
    tokens = lemmatize(tokens)
    return tokens

# Function to perform LDA topic modeling
def perform_lda(text_data):
    # Split text data into documents
    documents = text_data.strip().split('\n')

    # Preprocess documents
    print("Preprocessing documents...")
    processed_docs = [preprocess_text(doc) for doc in documents if doc.strip()]
    print(f"Number of processed documents: {len(processed_docs)}")

    # Check sample processed documents
    print("\nSample processed documents:")
    for doc in processed_docs[:5]:
        print(doc)

    # Create dictionary and corpus for LDA
    print("\nCreating dictionary and corpus...")
    dictionary = corpora.Dictionary(processed_docs)

    # Check dictionary size before filtering
    print(f"Dictionary size before filtering: {len(dictionary)}")

    # Adjust filter extremes parameters
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    print(f"Dictionary size after filtering: {len(dictionary)}")

    # Create Bag-of-Words corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    print(f"Number of documents in corpus: {len(corpus)}")

    # Check if corpus is empty
    if len(corpus) == 0 or len(dictionary) == 0:
        print("Error: The corpus or dictionary is empty. Adjust the preprocessing steps.")
        return

    # Determine the optimal number of topics using coherence score
    print("\nDetermining the optimal number of topics...")
    coherence_scores = []
    model_list = []
    for num_topics in range(2, 11):
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            workers=max(1, cpu_count() - 1),
            chunksize=2000,
            passes=10,
            random_state=100
        )
        model_list.append(lda_model)
        coherencemodel = CoherenceModel(
            model=lda_model,
            texts=processed_docs,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherencemodel.get_coherence()
        coherence_scores.append(coherence_score)
        print(f"Number of topics: {num_topics}, Coherence Score: {coherence_score:.4f}")

    # Select the model with the highest coherence score
    optimal_index = coherence_scores.index(max(coherence_scores))
    optimal_model = model_list[optimal_index]
    optimal_num_topics = optimal_index + 2  # Since range starts at 2
    print(f"\nOptimal number of topics: {optimal_num_topics}")

    # Print the topics discovered by the model
    print("\nTop words per topic:")
    for idx, topic in optimal_model.print_topics(-1):
        print(f"Topic {idx + 1}:")
        print(", ".join([word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]))
        print()

    # Visualize the topics using pyLDAvis
    try:
        print("Generating interactive topic visualization...")
        vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus, dictionary)
        save_path = os.path.join(os.path.expanduser("~"), "Downloads", "lda_visualization.html")
        pyLDAvis.save_html(vis, save_path)
        print(f"LDA visualization saved at: {save_path}")
    except Exception as e:
        print(f"An error occurred while generating visualization: {e}")

    # Plot coherence scores
    print("Plotting coherence scores...")
    x = range(2, 11)
    plt.figure(figsize=(10, 6))
    plt.plot(x, coherence_scores, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Scores by Number of Topics")
    plt.xticks(x)
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Path to your descriptions.txt file
    file_path = '/Users/flewolf/Downloads/descriptions.txt'

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist. Please check the path.")
    else:
        # Read the text data
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()

        # Optionally, sample a subset of the data for quicker testing
        # Uncomment the following line to use only the first 10,000 lines
        # text_data = '\n'.join(text_data.split('\n')[:10000])

        # Perform LDA topic modeling
        perform_lda(text_data)