AI Resume Analyzer
Smarter Resumes. Faster Hiring.

A production-ready AI-powered resume screening and analysis system that leverages state-of-the-art BERT models to automate recruitment workflows. Built for HR professionals, recruiters, and hiring managers who need intelligent candidate screening at scale.

🚀 Features
Core AI Capabilities
1.	🤖 BERT-Powered Classification: 24-category job classification with 93%+ accuracy
2.	🔍 Semantic Job Matching: Advanced similarity scoring using transformer embeddings
3.	⚡ GPU Acceleration: 10-20x faster training and inference with CUDA support
4.	📊 ATS Compatibility Scoring: Industry-standard applicant tracking system evaluation
5.	🎯 Intelligent Screening Pipeline: Multi-factor candidate assessment algorithm

Professional Web Interface
1.	🎨 Modern UI/UX: Glassmorphism design with gradient backgrounds and smooth animations
2.	📱 Responsive Design: Mobile-first approach with collapsible navigation
3.	📈 Interactive Charts: Real-time Plotly visualizations for scores and metrics
4.	🏷️ Smart Badges: Gamified strengths/weaknesses analysis with tooltips
5.	⚡ Real-time Processing: Cached models for instant analysis results

Enterprise Features
1.	📦 Batch Processing: Screen hundreds of resumes simultaneously
2.	📋 Detailed Reports: Exportable CSV reports with comprehensive candidate insights
3.	🔄 Production Ready: Scalable architecture with error handling and optimization
4.	🛡️ Robust Pipeline: End-to-end ML workflow from PDF extraction to hiring recommendations

🛠️ Tech Stack
1.	Component	Technology
2.	Frontend	Streamlit, HTML5, CSS3, JavaScript
3.	Backend	Python 3.8+, PyTorch 2.0+
4.	AI/ML	BERT (Transformers), Scikit-learn, CUDA
5.	Visualization	Plotly, Pandas
6.	Document Processing	PyPDF2, NLTK
7.	Deployment	Streamlit Cloud, Docker Ready
📦 Installation
Prerequisites 
1.	Python 3.8 or higher
2.	CUDA-compatible GPU (optional, for acceleration)
3.	8GB+ RAM recommended

Quick Start
Clone the repository

bash
git clone https://github.com/Swashinie/ai-resume-analyzer.git
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
1.	Open your browser to http://localhost:8501
2.	Upload a PDF resume and job description
3.	Get instant AI-powered analysis!

Structure

<img width="914" height="540" alt="image" src="https://github.com/user-attachments/assets/9380317e-825f-431b-a6d3-fc3aa01196ae" />


🎮 Usage Guide
Single Resume Analysis
1.	Navigate to the Upload tab
2.	Upload a PDF resume (drag & drop supported)
3.	Paste the target job description
4.	Click Analyze for instant results
5.	Review ATS score, category prediction, and recommendations

Batch Processing
1.	Go to Reports tab
2.	Upload multiple PDF resumes
3.	Provide a single job description for comparison
4.	Download comprehensive screening report as CSV

Key Metrics Explained
1.	ATS Score: Overall compatibility (0-100%)
2.	Category Confidence: Model certainty in job classification
3.	Semantic Similarity: Resume-job description match using AI
4.	Screening Score: Weighted final recommendation score

📊 Performance Benchmarks
1.	Metric	Value
2.	Training Dataset	2,483 resumes across 24 categories
3.	Classification Accuracy	93.08% on test data
4.	Average Processing Time	<2 seconds per resume (GPU)
5.	Similarity Scoring Range	0.599 - 0.758 (semantic matching)
6.	Supported File Types	PDF (text-extractable)

🎯 Use Cases
1.	HR Departments: Automate initial resume screening and candidate ranking
2.	Recruitment Agencies: Scale candidate evaluation across multiple clients
3.	Job Seekers: Optimize resumes for specific job descriptions and ATS systems
4.	Career Services: Provide data-driven feedback to students and professionals
5.	Hiring Managers: Get AI-powered insights for faster decision making

🚀 Deployment
1.	Streamlit Cloud (Recommended)
2.	Push code to GitHub repository
3.	Connect to share.streamlit.io
4.	Deploy with one click
5.	Share public URL with your team

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
1.	Fork the repository
2.	Create a feature branch (git checkout -b feature/amazing-feature)
3.	Make your changes
4.	Add tests for new functionality
5.	Commit changes (git commit -am 'Add amazing feature')
6.	Push to branch (git push origin feature/amazing-feature)
7.	Open a Pull Request

📈 Roadmap
1.	Multi-language Support: Expand beyond English resumes
2.	Advanced Analytics: Industry benchmarking and trends
3.	API Integration: REST API for enterprise systems
4.	Enhanced NLP: Fine-tuned models for specific industries
5.	Real-time Collaboration: Multi-user screening workflows
6.	Mobile App: Native iOS/Android applications

🐛 Known Issues & Limitations
	PDF text extraction quality depends on document formatting
	BERT model performance varies by industry-specific terminology
	GPU memory requirements scale with batch processing size
	Model retraining required for new job categories

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
1.	Hugging Face Transformers - BERT model implementation
2.	Streamlit Team - Excellent web app framework
3.	PyTorch Community - Deep learning infrastructure
4.	Contributors - Everyone who helped improve this project

📞 Support
	Documentation: Wiki
	Issues: GitHub Issues
	Discussions: GitHub Discussions
	Email: swashinier@gmail.com

<div align="center">
Built with ❤️ by SWASHINIE ☺️

⭐ Star this repo if it helped you! ⭐

</div>

