import pandas as pd
from screening_pipeline import screen_resume
import os

def batch_screen_resumes():
    """Process multiple resumes against job descriptions"""
    try:
        # Load resume data (correct path)
        resume_df = pd.read_csv('data/resume_texts.csv')
        job_df = pd.read_csv('data/job_descriptions.csv')
        
        results = []
        
        print(f"Processing {len(resume_df)} resumes against {len(job_df)} jobs...")
        
        # Process first 5 resumes against all jobs
        for i, resume_text in enumerate(resume_df['resume_text'][:5]):
            best_score = 0
            best_job = None
            
            print(f"Processing resume {i+1}/5...")
            
            # Test against all job descriptions
            for j, job_desc in enumerate(job_df['cleaned_description']):
                result = screen_resume(resume_text, job_desc)
                
                if result['screening_score'] > best_score:
                    best_score = result['screening_score']
                    best_job = {
                        'job_title': job_df.iloc[j]['job_title'],
                        'job_index': j,
                        **result
                    }
            
            results.append({
                'resume_index': i,
                'resume_category': resume_df.iloc[i]['category'],
                'best_matching_job': best_job['job_title'],
                'screening_score': best_job['screening_score'],
                'job_similarity': best_job['job_similarity'],
                'category_confidence': best_job['category_confidence'],
                'match_quality': 'Strong' if best_job['screening_score'] >= 0.7 else 'Moderate'
            })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        results_df.to_csv('data/screening_results.csv', index=False)
        
        print("\n=== Batch Screening Complete ===")
        print(results_df)
        print(f"\nResults saved to: data/screening_results.csv")
        
        return results_df
        
    except FileNotFoundError as e:
        print(f"Error: Required CSV file not found - {e}")
        print("Please ensure these files exist:")
        print("- data/job_descriptions.csv")
        print("- data/resume_texts.csv")

if __name__ == "__main__":
    batch_screen_resumes()
