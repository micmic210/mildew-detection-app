# Mildew Detection in Cherry Leaves

## Introduction 

Introduction

This project focuses on creating a machine learning solution to address the challenge of powdery mildew outbreaks in cherry plantations at Farmy & Foods. Currently, the process of manually inspecting each tree is time-intensive and labor-intensive, making it inefficient for large-scale operations. By leveraging image analysis and deep learning, this project aims to develop an efficient and accurate system for detecting powdery mildew in cherry leaves.

The project will integrate a robust machine learning model with a user-friendly web application to:
- Automate inspections: Reduce manual inspection time and associated labor costs.
- Enhance disease management: Enable early detection and intervention to minimize crop losses.
- Increase operational efficiency and sustainability: Support more efficient, scalable, and environmentally sustainable agricultural practices at Farmy & Foods.

The solution will leverage deep learning techniques, specifically Convolutional Neural Networks (CNNs), to analyze images of cherry leaves and classify them as healthy or infected with powdery mildew. By using real-world datasets of cherry leaves collected from Farmy & Foods’ plantations, the system will be tailored to their specific needs while offering the potential for scalability to other crops in the future.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

Farmy & Foods is experiencing a powdery mildew outbreak in their cherry plantations.
- Current Challenges:
	- Manual inspection of each tree is time-consuming and labor-intensive.
	- Employees spend significant time visually inspecting leaves for signs of mildew, taking approximately 30 minutes per tree.
	- The manual process is unsustainable for thousands of trees spread across multiple farms nationwide.
- Proposed Solution:
	- Implement a machine learning (ML) system to instantly and accurately detect powdery mildew in cherry leaves using image analysis.
	- If successful, this system can be scaled to other crops and pest/disease detection processes.
- Client Objectives:
	1.	Visual Differentiation Study: Research and document the visual characteristics of healthy vs. infected cherry leaves to better understand the disease.
	2.	Predictive Model: Develop an ML model to automate the detection of powdery mildew, significantly reducing the time and effort required for inspections.

In essence, Farmy & Foods requires a scalable and efficient solution to combat powdery mildew in their cherry plantations. By automating the detection process, the ML model will save time, reduce manual labor, and provide consistent and accurate results, enabling the company to better manage their crops and resources.

## Hypothesis and how to validate?
Hypothesis 1: Visual Distinction
- Hypothesis: Healthy leaves and mildew-infected leaves exhibit distinct visual patterns that can be identified through data analysis.
- Validation: Compute average and variability images for both healthy and infected leaves. Use montages to highlight these differences visually.

Hypothesis 2: Classification Accuracy
- Hypothesis: A CNN can classify cherry leaves as healthy or infected with at least 97% accuracy.
- Validation: Train the CNN model and evaluate its performance using precision, recall, F1-score, and accuracy metrics on a test dataset.


## The rationale to map the business requirements to the Data Visualisations and ML tasks

1.	Visual Differentiation:
    - Data Visualization: Create average and variability images for healthy and infected leaves.
	- Rationale: Helps stakeholders visually identify distinct patterns and characteristics of mildew-infected leaves.
2.	Disease Detection:
	- ML Task: Train a CNN model for binary classification (Healthy vs. Infected).
	- Rationale: Automates the manual inspection process, ensuring consistent and accurate results with minimal time investment.
3.	Farm Insights:
	- Data Visualization: Generate heatmaps and bar charts for infection trends, severity levels, and geographical distributions.
	- Rationale: Provides actionable insights to prioritize treatment and optimize resource allocation.

## ML Business Case
The objective of this project is to develop a machine learning pipeline capable of automating the detection of powdery mildew in cherry leaves, achieving the following goals:
- 1 - Automate Inspection:
	- Reduce the manual inspection time from 30 minutes per tree to a matter of seconds through an automated classification system.
- 2 - Provide Actionable Insights:
	- Generate farm-wide metrics to identify infection trends, high-risk areas, and prioritize treatment allocation for improved operational efficiency.

Key Goals:
- Achieve at least 97% accuracy for leaf classification.
- Provide real-time predictions and insights to reduce inspection time and improve farm management.


Machine Learning Pipeline: Steps and Business Relevance

The machine learning pipeline is designed to address the business requirements of Visual Differentiation, Disease Detection, and Farm Insights. Each step in the pipeline contributes to the automation and efficiency goals set by Farmy & Foods:

1. Data Collection
    - Details: 
        - The dataset contains over 4,000 images of cherry leaves labeled as “Healthy” or “Infected with Powdery Mildew.” This data was sourced from Kaggle and reflects the real-world challenges faced by Farmy & Foods.
	- Business Relevance: 
        - Supports Visual Differentiation by providing a comprehensive dataset for identifying patterns in healthy and infected leaves.
	    - Lays the foundation for Disease Detection and Farm Insights by ensuring a balanced dataset for robust model training and accurate predictions.

2. Data Preprocessing
	- Details:
	    - Images are resized to a uniform size (50x50 pixels) for model consistency.
	    - Pixel values are normalized to enhance model performance.
	    - Data augmentation techniques (e.g., rotation, flipping, cropping) are applied to increase dataset diversity and improve model generalization.
	- Business Relevance:
	    - Ensures Disease Detection accuracy by preparing high-quality input data for the model.
	    - Facilitates Visual Differentiation by standardizing and enhancing image clarity, making patterns more identifiable.

3. Feature Extraction
	- Details:
        - Convolutional Neural Networks (CNNs) are utilized to automatically extract features from the images, such as edges, textures, and patterns that distinguish healthy and infected leaves.
	- Business Relevance:
	    - Critical for Visual Differentiation as it identifies the unique characteristics of healthy vs. infected leaves.
	    - Forms the basis of Disease Detection, enabling the model to classify leaves with high accuracy.

4. Model Selection
	- Details:
	    - A CNN architecture is implemented, including:
	    - Convolutional and pooling layers for feature extraction.
	    - Fully connected layers for classification.
	    - An output layer with a sigmoid activation function for binary classification (Healthy/Infected).
	- Business Relevance:
	    - Drives Disease Detection by ensuring that the model can reliably predict leaf health status.
	    - Supports scalability, allowing the solution to handle large volumes of image data from multiple farms.

5. Model Training
	- Details:
	    - The dataset is split into training, validation, and test subsets to ensure unbiased evaluation.
	    - The model is trained using:
	    - The Adam optimizer for faster convergence.
	    - Binary cross-entropy as the loss function.
	    - Early stopping to prevent overfitting.
	- Business Relevance:
	    - Enhances Disease Detection accuracy and reliability, ensuring the model meets the 97% accuracy target.
	    - Supports Farm Insights by ensuring consistent predictions for aggregated metrics and heatmaps.

6. Model Evaluation
	- Details:
	    - Performance is evaluated using metrics such as:
	    - Accuracy
	    - Precision
	    - Recall
	    - F1-score
	    - The model’s robustness is validated using unseen test data.
	- Business Relevance:
	    - Provides confidence in Disease Detection predictions, ensuring stakeholders can rely on the model for operational decisions.
	    - Builds trust in Farm Insights, enabling effective resource allocation and treatment prioritization.

7. Deployment
	- Details:
	    - The trained model is saved as a .h5 file and deployed using Streamlit for real-time predictions.
	    - A user-friendly dashboard allows clients to upload images and view results instantly.
	- Business Relevance:
	    - Fulfills Disease Detection by enabling real-time predictions for healthy vs. infected leaves.
	    - Forms the backbone of Farm Insights, allowing stakeholders to visualize infection trends and metrics through the dashboard.

8. Monitoring and Maintenance
	- Details:
	    - The model is periodically retrained with new data to maintain accuracy as environmental conditions change.
	    - Prediction performance is continuously monitored in production.
	- Business Relevance:
	    - Ensures long-term reliability of Disease Detection and Farm Insights, enabling the system to adapt to evolving challenges.

Summary of Business Case Impact

This pipeline directly addresses the challenges faced by Farmy & Foods, automating the manual inspection process and providing actionable insights. By combining Visual Differentiation, Disease Detection, and Farm Insights, the solution delivers both immediate and long-term value:
- Efficiency: Reduces inspection time from 30 minutes per tree to seconds.
- Accuracy: Provides reliable predictions with a target accuracy of 97%.
- Insights: Generates farm-wide metrics, enabling smarter resource allocation and effective disease management.

## CRISP-DM Framework

1.	Business Understanding:
	- Problem: Manual inspection of cherry leaves is inefficient and unscalable.
	- Goal: Develop a scalable ML solution for leaf classification and farm insights.
2.	Data Understanding:
	-	Dataset sourced from Kaggle, containing 4,000 labeled images.
	-	Images represent two classes: Healthy and Infected.
3.	Data Preparation:
	-	Resize images to 50x50 pixels for model compatibility.
	-	Normalize pixel values and augment data (rotation, flipping) for improved generalization.
4.	Modeling:
	-	Build a CNN architecture with convolutional, pooling, and fully connected layers.
5.	Evaluation:
	-	Evaluate model performance using metrics such as precision, recall, F1-score, and accuracy.
6.	Deployment:
	-	Deploy the model within a Streamlit dashboard for real-time predictions and insights.


## Dashboard Design
The dashboard consists of five interactive pages:
1.	Project Overview:
	-	Introduces powdery mildew and project objectives.
	-	Displays links to the README and dataset.
2.	Leaf Visualizer:
	-	Average and variability images for healthy and infected leaves.
	-	Image montage comparisons.
3.	Mildew Detector:
	-	File uploader for real-time leaf classification.
	-	Option to download prediction results as a CSV file.
4.	Farm Insights:
	-	Aggregated metrics: Infection percentages, severity levels.
	-	Heatmap visualizing infection patterns across farms.
	-	Filters for date range and farm region.
5.	Model Performance:
	-	Training and validation accuracy/loss plots.
	-	Confusion matrix and classification report.

## User Stories

1.	Intuitive Navigation
	-	User Story:
    As a client, I want an intuitive dashboard with clear navigation so that I can easily access data, predictions, and insights.
	•	Acceptance Criteria:
	•	A navigation bar is present and allows switching between all pages.
	•	All navigation links are clearly labeled and functional.
	•	The user can access any page in no more than two clicks.

2.	Visual Differentiation
	-	User Story:
As a client, I want to observe average and variability images of healthy and mildew-infected cherry leaves so that I can visually differentiate between the two categories.
	•	Acceptance Criteria:
	•	The dashboard displays average and variability images for healthy and infected leaves.
	•	The user can toggle between these visualizations using checkboxes or buttons.
	•	The visualizations are clear and labeled with appropriate captions.

3.	Image Montage
	-	User Story:
As a client, I want to view montages of healthy and infected leaves so that I can compare them more easily.
	•	Acceptance Criteria:
	•	The user can select “Healthy” or “Infected” leaves to create a montage.
	•	The montage displays at least 9 images per category in a grid format.
	•	There is a button to dynamically generate a new montage.

4.	Real-Time Predictions
	-	User Story:
As a client, I want to upload images of cherry leaves and receive predictions about their health status (healthy/infected) in real-time.
	•	Acceptance Criteria:
	•	A file uploader is available and supports single and multiple image uploads.
	•	The system predicts the health status of each uploaded image with at least 97% accuracy.
	•	Predictions are displayed on the dashboard with confidence scores.

5.	CSV Download of Predictions
	-	User Story:
As a client, I want to download a CSV report summarizing predictions for all uploaded images so that I can keep a record for future analysis.
	•	Acceptance Criteria:
	•	A download button is available on the prediction results page.
	•	The CSV includes the filename, prediction, and confidence score for each image.
	•	The file downloads correctly when clicked.

6.	Infection Rate Summary
	-	User Story:
As a client, I want to see a summary of the infection rate (percentage of healthy vs. infected leaves) based on the uploaded images so that I can quickly understand the overall situation.
	•	Acceptance Criteria:
	•	A pie chart or bar chart displays the percentage of healthy vs. infected leaves.
	•	The chart updates dynamically based on the uploaded images.
	•	The chart is labeled clearly and easy to interpret.


## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media



## Acknowledgements

- 
