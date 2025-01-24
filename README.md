# Mildew Detection in Cherry Leaves

## Project Overview

Farmy & Foods faces a significant challenge with powdery mildew, a fungal disease that impacts cherry leaves. This disease appears as white, powdery spots and can significantly reduce crop yield and quality. Currently, the manual inspection process for identifying mildew is labor-intensive, requiring 30 minutes per tree and making it unscalable for thousands of trees spread across multiple farms.

The goal of this project is to leverage machine learning to automate the detection of powdery mildew in cherry leaves using image classification. The solution will provide real-time predictions and farm-wide insights, enabling stakeholders to prioritize treatment efficiently and reduce operational costs.


## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements
1.	Visual Differentiation:
	- Conduct a study to visually differentiate healthy cherry leaves from those with powdery mildew using data visualization techniques.
2.	Disease Detection:
	- Develop a Convolutional Neural Network (CNN) to classify cherry leaf images as healthy or infected.
3.	Farm Insights:
	- Provide farm-wide aggregated metrics, including:
	- Percentage of infected vs. healthy leaves.
	- Heatmaps showing infection distribution across farm regions.
	- Severity breakdowns for prioritizing treatment.

## Hypothesis and how to validate?
Hypothesis 1: Visual Distinction
- Hypothesis: Healthy leaves and mildew-infected leaves exhibit distinct visual patterns that can be identified through data analysis.
- Validation: Compute average and variability images for both healthy and infected leaves. Use montages to highlight these differences visually.


Hypothesis 2: Classification Accuracy
- Hypothesis: A CNN can classify cherry leaves as healthy or infected with at least 97% accuracy.
- Validation: Train the CNN model and evaluate its performance using precision, recall, F1-score, and accuracy metrics on a test dataset.

Hypothesis 3: Farm-Level Insights
- Hypothesis: Aggregating predictions across multiple leaves can reveal infection patterns and high-risk areas within the farm.
- Validation: Generate aggregated farm-wide metrics, including infection rates, severity distributions, and heatmaps, to visualize infection trends.


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
