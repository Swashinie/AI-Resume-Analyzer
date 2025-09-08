**AI Resume Analyzer**_
_Smarter Resumes. Faster Hiring._

A production-ready AI-powered resume screening and analysis system that leverages state-of-the-art BERT models to automate recruitment workflows. Built for HR professionals, recruiters, and hiring managers who need intelligent candidate screening at scale.

🚀 **Features**
Core AI Capabilities
🤖 BERT-Powered Classification: 24-category job classification with 93%+ accuracy

🔍 Semantic Job Matching: Advanced similarity scoring using transformer embeddings

⚡ GPU Acceleration: 10-20x faster training and inference with CUDA support

📊 ATS Compatibility Scoring: Industry-standard applicant tracking system evaluation

🎯 Intelligent Screening Pipeline: Multi-factor candidate assessment algorithm

**Professional Web Interface**
🎨 Modern UI/UX: Glassmorphism design with gradient backgrounds and smooth animations

📱 Responsive Design: Mobile-first approach with collapsible navigation

📈 Interactive Charts: Real-time Plotly visualizations for scores and metrics

🏷️ Smart Badges: Gamified strengths/weaknesses analysis with tooltips

⚡ Real-time Processing: Cached models for instant analysis results

**Enterprise Features**
📦 Batch Processing: Screen hundreds of resumes simultaneously

📋 Detailed Reports: Exportable CSV reports with comprehensive candidate insights

🔄 Production Ready: Scalable architecture with error handling and optimization

🛡️ Robust Pipeline: End-to-end ML workflow from PDF extraction to hiring recommendations

🛠️ Tech Stack
Component	Technology
Frontend	Streamlit, HTML5, CSS3, JavaScript
Backend	Python 3.8+, PyTorch 2.0+
AI/ML	BERT (Transformers), Scikit-learn, CUDA
Visualization	Plotly, Pandas
Document Processing	PyPDF2, NLTK
Deployment	Streamlit Cloud, Docker Ready
📦 Installation
Prerequisites
Python 3.8 or higher

CUDA-compatible GPU (optional, for acceleration)

8GB+ RAM recommended

Quick Start
Clone the repository

bash
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer
Create virtual environment

bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux  
source .venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Download pre-trained models

bash
# Models will be automatically downloaded on first run
# Or manually place your trained models in /models/
Run the application

bash
streamlit run web_app.py
Access the app

Open your browser to http://localhost:8501

Upload a PDF resume and job description

Get instant AI-powered analysis!

🏗️ Project Structure
text
ai-resume-analyzer/
├── 📁 data/                     # Training data and job descriptions
│   ├── resume_texts.csv         # Processed resume dataset (2,483+ entries)
│   ├── job_descriptions.csv     # Job posting database
│   └── screening_results.csv    # Batch processing outputs
├── 📁 models/                   # Trained AI models
│   ├── best_model.pth          # BERT classifier (24 categories)
│   └── label_encoder.pkl       # Category label mappings
├── 📁 utils/                    # Core utilities
│   ├── dataset.py              # PyTorch dataset classes
│   ├── similarity_utils.py     # BERT embedding functions
│   └── preprocessing.py        # Text processing pipeline
├── 📄 web_app.py               # Main Streamlit application
├── 📄 train.py                 # Model training script
├── 📄 screening_pipeline.py    # Core analysis pipeline
├── 📄 batch_processor.py       # Bulk resume processing
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md               # Project documentation
🎮 Usage Guide
Single Resume Analysis
Navigate to the Upload tab

Upload a PDF resume (drag & drop supported)

Paste the target job description

Click Analyze for instant results

Review ATS score, category prediction, and recommendations

Batch Processing
Go to Reports tab

Upload multiple PDF resumes

Provide a single job description for comparison

Download comprehensive screening report as CSV

Key Metrics Explained
ATS Score: Overall compatibility (0-100%)

Category Confidence: Model certainty in job classification

Semantic Similarity: Resume-job description match using AI

Screening Score: Weighted final recommendation score

📊 Performance Benchmarks
Metric	Value
Training Dataset	2,483 resumes across 24 categories
Classification Accuracy	93.08% on test data
Average Processing Time	<2 seconds per resume (GPU)
Similarity Scoring Range	0.599 - 0.758 (semantic matching)
Supported File Types	PDF (text-extractable)
🎯 Use Cases
HR Departments: Automate initial resume screening and candidate ranking

Recruitment Agencies: Scale candidate evaluation across multiple clients

Job Seekers: Optimize resumes for specific job descriptions and ATS systems

Career Services: Provide data-driven feedback to students and professionals

Hiring Managers: Get AI-powered insights for faster decision making

🚀 Deployment
Streamlit Cloud (Recommended)
Push code to GitHub repository

Connect to share.streamlit.io

Deploy with one click

Share public URL with your team

Docker Deployment
bash
# Build container
docker build -t ai-resume-analyzer .

# Run application
docker run -p 8501:8501 ai-resume-analyzer
Local Production
bash
# Install production dependencies
pip install -r requirements-prod.txt

# Run with optimized settings
streamlit run web_app.py --server.enableCORS false --server.enableXsrfProtection false
🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

Development Setup
Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Make your changes

Add tests for new functionality

Commit changes (git commit -am 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open a Pull Request

📈 Roadmap
 Multi-language Support: Expand beyond English resumes

 Advanced Analytics: Industry benchmarking and trends

 API Integration: REST API for enterprise systems

 Enhanced NLP: Fine-tuned models for specific industries

 Real-time Collaboration: Multi-user screening workflows

 Mobile App: Native iOS/Android applications

🐛 Known Issues & Limitations
PDF text extraction quality depends on document formatting

BERT model performance varies by industry-specific terminology

GPU memory requirements scale with batch processing size

Model retraining required for new job categories

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Hugging Face Transformers - BERT model implementation

Streamlit Team - Excellent web app framework

PyTorch Community - Deep learning infrastructure

Contributors - Everyone who helped improve this project

📞 Support
Documentation: Wiki

Issues: GitHub Issues

Discussions: GitHub Discussions

Email: your.email@domain.com

<div align="center">
Built with ❤️ by [Your Name]

⭐ Star this repo if it helped you! ⭐

</div>
This project demonstrates the power of AI in transforming recruitment workflows. From PDF extraction to intelligent candidate ranking, every component is designed for production use at enterprise scale.
