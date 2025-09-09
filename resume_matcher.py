from utils.similarity_utils import get_bert_embeddings, compute_similarity
import pandas as pd

def main():
    try:
        # Load job descriptions (correct path)
        job_df = pd.read_csv('data/job_descriptions.csv')
        
        # Load resume data (correct path)
        resume_df = pd.read_csv('data/resume_texts.csv')
        resume_texts = resume_df['resume_text'].tolist()
        
        # Get embeddings for resumes and jobs (test with first 10 resumes)
        print("Generating embeddings...")
        resume_embeddings = get_bert_embeddings(resume_texts[:10])  
        job_embeddings = get_bert_embeddings(job_df['cleaned_description'].tolist())
        
        print(f"\nMatching {len(resume_embeddings)} resumes against {len(job_embeddings)} jobs...\n")
        
        # Find best matches for each resume
        for i, resume_emb in enumerate(resume_embeddings):
            similarities = compute_similarity(resume_emb, job_embeddings)
            best_job_idx = similarities.argmax()
            
            print(f"Resume {i+1} best matches Job {best_job_idx+1}")
            print(f"Job Title: {job_df.iloc[best_job_idx]['job_title']}")
            print(f"Similarity Score: {similarities[best_job_idx]:.3f}")
            print(f"Resume Category: {resume_df.iloc[i]['category']}")
            print("-" * 50)
            
    except FileNotFoundError as e:
        print(f"Error: Required CSV file not found - {e}")
        print("Please ensure these files exist:")
        print("- data/job_descriptions.csv")
        print("- data/resume_texts.csv")

if __name__ == "__main__":
    main()
