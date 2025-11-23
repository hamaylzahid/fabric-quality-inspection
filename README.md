<br><h1 align="center">Automated Defect Detection System</h1><br>
<p p align="center" style="color: #555; font-size: 16px;">
  A deep learning system for detecting and classifying fabric defects.<br>
  This project uses a fine-tuned <strong>ResNet18</strong> model to identify defects such as <strong>holes</strong>, <strong>vertical</strong>, and <strong>horizontal</strong> patterns.<br>
  The system validates model performance and provides visual predictions for real-world fabric inspection.
</p>

<!-- Core Technology Stack -->
<h4 align="center">Core Technology Stack</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python&logoColor=white" alt="Python Badge" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=flat&logo=pytorch&logoColor=white" alt="PyTorch Badge" />
  <img src="https://img.shields.io/badge/Torchvision-ML-orange?style=flat&logo=opencv&logoColor=white" alt="Torchvision Badge" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-learn Badge" />
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c?style=flat&logo=plotly&logoColor=white" alt="Matplotlib Badge" />
</p>

<!-- Project Info -->
<h4 align="center">Project Info</h4>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/hamaylzahid/automated-defect-detection?style=flat&color=orange&logo=github" alt="Last Commit Badge" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=flat&logo=vercel&logoColor=white" alt="Status Badge" />
</p>

<!-- Dataset Section -->
<br><h2 align="center">Dataset</h2><br>
<p>
  The system is trained on the <strong>Fabric Defects Dataset</strong> from Kaggle, containing multiple fabric images labeled as <strong>hole</strong>, <strong>vertical</strong>, and <strong>horizontal</strong> defects.
</p>
<p>
  Images are preprocessed to <strong>224×224 pixels</strong> to match the input size for the ResNet18 model. The dataset is split into <strong>train</strong> and <strong>validation</strong> sets, with an additional check set for visual evaluation.
</p>
<p>
  <strong>License:</strong> MIT &nbsp; | &nbsp; <strong>Total Images:</strong> Varies per dataset &nbsp; | &nbsp; <strong>Classes:</strong> 3
</p><br>
<p align="center">
    <a href="https://www.kaggle.com/datasets/andrewmvd/fabric-defects" target="_blank">
        <img src="https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle&style=for-the-badge" 
             alt="Kaggle Dataset Badge">
    </a>
</p>

<hr style="border:1px solid #ccc; margin-top:20px; margin-bottom:20px;">

<h2 align="center">📖 Table of Contents</h2>
<ul style="list-style-type:none; font-size:16px; color:#444; text-align:center; line-height:2;">
  <li>🧠 <a href="#-project-overview">Project Overview</a></li>
  <li>🎯 <a href="#-core-objectives">Core Objectives</a></li>
  <li>🖼️ <a href="#-dataset-and-images">Dataset & Images</a></li>
  <li>🛠️ <a href="#system-architecture">System Architecture</a></li>
  <li>🧪 <a href="#-training-workflow">Training Workflow</a></li>
  <li>📊 <a href="#-evaluation--metrics">Evaluation & Metrics</a></li>
  <li>⚙️ <a href="#setup-installation">Setup & Installation</a></li>
  <li>🤝 <a href="#-contact--contribution">Contact & Contribution</a></li>
  <li>📜 <a href="#-license">License</a></li>
</ul>

<hr style="border:1px solid #ccc; margin-top:20px; margin-bottom:20px;">

<br><h2 align="center">🧠 Project Overview</h2><br>
<p>
  <strong>Automated Defect Detection System</strong> is a deep learning solution for detecting and classifying fabric defects in real-world scenarios. 
  Using <code>PyTorch</code> and <code>torchvision</code>, it leverages the ResNet18 architecture to extract robust features and produce reliable predictions.
</p>
<p>
  From raw fabric images to actionable defect identification, the system enables efficient quality control with high precision.
</p>
<p>
  <em>An end-to-end pipeline for automated defect detection and fabric quality inspection.</em>
</p>

<br><h2 align="center" id="core-objectives">🎯 Core Objectives</h2><br>
<ul><br>
  <li>Implement transfer learning with pre-trained ResNet18.</li>
  <li>Classify fabric images into three defect categories.</li>
  <li>Validate model performance using standard metrics.</li>
  <li>Provide visual predictions and inspection-ready outputs.</li>
  <li>Support deployment for real-time defect detection.</li>
</ul>

<br><h2 align="center" id="system-architecture">🛠️ System Architecture</h2><br>
<ul>
  <li>Load pre-trained ResNet18 from torchvision.</li>
  <li>Replace the final classification layer with 3 output classes.</li>
  <li>Freeze base layers, fine-tune the classification head.</li>
  <li>Train with CrossEntropyLoss and SGD optimizer.</li>
  <li>Evaluate and predict on validation and real-world fabric images.</li>
</ul>

<br><h2 align="center" id="training-workflow">🧪 Training Workflow</h2><br>
<ul>
  <li>Preprocess images: resize, normalize, and apply augmentations.</li>
  <li>Fine-tune top layers of ResNet18 for a defined number of epochs.</li>
  <li>Monitor performance using accuracy, loss, and confusion matrices.</li>
  <li>Test predictions on unseen images for visual evaluation.</li>
</ul>

<br><h2 align="center" id="evaluation-metrics">📊 Evaluation & Metrics</h2><br>
<ul style="text-align:left; margin-left:40px;">
    <li>Precision, Recall, F1-Score for each defect type.</li>
    <li>Confusion matrix to analyze misclassifications.</li>
    <li>Real-world predictions verified against actual defects.</li>
</ul>
---
<br><h2 align="center" id="visual-results">🖼️ Visual Results</h2><br>

<br><h3 align="center"> Utility Function for Predicting a Single Image</h3><br>
<div align="center">
  <img src="visuals/Utility Function for Predicting a Single Image.png" alt="Single Image Prediction" width="600">
</div>
<p align="center"><br>
  This screenshot shows the utility function in action: a single fabric image is input into the trained ResNet18 model, and the system predicts the defect type with confidence scores. 
  It demonstrates how users can easily test individual images for quick inspection.
</p>


<br><h3 align="center"> Visualize Model Predictions on Test Images</h3><br>

<div align="center">
  <img src="visuals/Visualize Model Predictions on Test Images.png" alt="Model Predictions" width="600">
</div><br>
<p align="center">
  This visualization shows the model's predictions on a batch of test images. Each image is labeled with the predicted defect type, enabling a clear assessment of model performance in real-world conditions. 
  It demonstrates the pipeline's ability to handle multiple images efficiently and provide actionable insights.
</p>

<br><h3 align="center"> Visualize Misclassified Samples</h3><br>

<div align="center">
  <img src="visuals/Visualize Misclassified Samples.png" alt="Misclassified Samples" width="600">
</div>
<p align="center">
  This screenshot highlights misclassified images from the validation set. The system shows both the predicted label and the actual label, helping users analyze errors and understand model limitations. 
  Misclassification visualization is crucial for improving model accuracy and reliability.
</p><br>
<br><h2 align="center" id="setup-installation">⚙️ Setup & Installation</h2><br>
<pre>
# Clone repository
git clone https://github.com/hamaylzahid/automated-defect-detection.git

# Navigate to folder
cd automated-defect-detection

# Install dependencies
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook Defect_detection_system.ipynb
</pre>

<h2 align="center">🤝 Contact & Contribution</h2>

<p align="center">
  Have feedback, want to collaborate, or just say hello?<br>
  <strong>Let’s connect and improve automated defect detection together.</strong>
</p>

<p align="center">
  📬 <a href="mailto:maylzahid588@gmail.com">maylzahid588@gmail.com</a> &nbsp; | &nbsp;
  💼 <a href="https://www.linkedin.com/in/your-linkedin-profile">LinkedIn</a> &nbsp; | &nbsp;
  🌐 <a href="https://github.com/hamaylzahid/automated-defect-detection">GitHub Repo</a>
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid/automated-defect-detection/stargazers">
    <img src="https://img.shields.io/badge/Star%20This%20Project-Give%20a%20Star-yellow?style=for-the-badge&logo=github" alt="Star Badge" />
  </a>
  <a href="https://github.com/hamaylzahid/automated-defect-detection/pulls">
    <img src="https://img.shields.io/badge/Contribute-Pull%20Requests%20Welcome-2ea44f?style=for-the-badge&logo=github" alt="PRs Welcome Badge" />
  </a>
</p>

<p align="center">
  ⭐ Found this project helpful? Give it a star on GitHub!<br>
  🤝 Want to improve it? Submit a PR and join the mission.<br>
  <sub><i>Your contributions help enhance real-world defect detection systems.</i></sub>
</p>

<br>
<h2 align="center">📜 License</h2>

<p align="center">
  <a href="https://github.com/hamaylzahid/automated-defect-detection/commits/main">
    <img src="https://img.shields.io/badge/Last%20Commit-Recent-blue" alt="Last Commit">
  </a>
  <a href="https://github.com/hamaylzahid/automated-defect-detection">
    <img src="https://img.shields.io/badge/Repo%20Size-Moderate-lightgrey" alt="Repo Size">
  </a>
</p>

<p align="center">
  This project is licensed under the <strong>MIT License</strong> — free to use, modify, and expand.
</p>

<p align="center">
  ✅ <strong>Project Status:</strong> Complete & Portfolio-Ready<br>
  🧾 <strong>License:</strong> MIT — <a href="LICENSE">View License »</a>
</p>

<p align="center">
  <strong>Crafted with deep learning expertise & real-world defect detection applications</strong> 🧵✨
</p>

<p align="center">
  <a href="https://github.com/hamaylzahid">
    <img src="https://img.shields.io/badge/GitHub-%40hamaylzahid-181717?style=flat-square&logo=github" alt="GitHub" />
  </a>
  •
  <a href="mailto:maylzahid588@gmail.com">
    <img src="https://img.shields.io/badge/Email-Contact%20Me-red?style=flat-square&logo=gmail&logoColor=white" alt="Email" />
  </a>
  •
  <a href="https://github.com/hamaylzahid/automated-defect-detection">
    <img src="https://img.shields.io/badge/Repo-Link-blueviolet?style=flat-square&logo=github" alt="Repo" />
  </a>
  <br>
  <a href="https://github.com/hamaylzahid/automated-defect-detection/fork">
    <img src="https://img.shields.io/badge/Fork%20This%20Project-Contribute%20to%20AI-2ea44f?style=flat-square&logo=github" alt="Fork Badge" />
  </a>
</p>

<p align="center">
  <sub><i>Designed for real-world fabric defect detection and deep learning showcase.</i></sub>
</p>

<p align="center">
  🤖 <b>Use this project to demonstrate your expertise in computer vision and AI</b><br>
  🧵 Clone it, modify it, expand it — and build real-world automated defect detection solutions.
</p>


