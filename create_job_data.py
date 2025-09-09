import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Create sample job descriptions
sample_jobs = {
    'job_title': [
        'Data Scientist',
        'Software Engineer', 
        'DevOps Engineer',
        'Web Developer',
        'Marketing Manager',
        'Financial Analyst',
        'HR Manager',
        'Sales Representative',
        'Graphic Designer',
        'Project Manager'
    ],
    'cleaned_description': [
        'Seeking experienced data scientist with Python, machine learning, and statistical analysis skills.',
        'Looking for software engineer with Java, React, and full-stack development experience.',
        'DevOps engineer needed with AWS, Docker, Kubernetes, and CI/CD pipeline expertise.',
        'Web developer position requires HTML, CSS, JavaScript, and modern framework knowledge.',
        'Marketing manager role involves digital marketing, campaign management, and analytics.',
        'Financial analyst position requires Excel, financial modeling, and data analysis skills.',
        'HR manager needed with recruitment, employee relations, and policy development experience.',
        'Sales representative role involves client acquisition, relationship management, and target achievement.',
        'Graphic designer position requires Adobe Creative Suite, branding, and visual design skills.',
        'Project manager role involves team coordination, timeline management, and stakeholder communication.'
    ]
}

# Create the DataFrame and save to CSV
job_df = pd.DataFrame(sample_jobs)
job_df.to_csv('data/job_descriptions.csv', index=False)

print("✅ Sample job descriptions created successfully!")
print(f"📁 File saved: data/job_descriptions.csv")
print(f"📊 Created {len(job_df)} job descriptions")
print("\nFirst few rows:")
print(job_df.head())
