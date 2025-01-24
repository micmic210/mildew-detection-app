# Mildew Detection in Cherry Leaves

## Project Overview

The cherry plantation at Farmy & Foods faces a significant challenge with powdery mildew, a fungal disease that affects the leaves of cherry trees. The disease manifests as white powdery spots on leaves, potentially leading to poor crop yield and reduced product quality.

Currently, employees spend approximately 30 minutes inspecting each tree for signs of mildew and applying treatments where necessary. With thousands of trees across multiple farms, this process is time-consuming and unsustainable.

This project leverages machine learning to build an automated mildew detection system. The system analyzes images of cherry leaves to instantly determine whether a leaf is healthy or infected, streamlining the inspection process and improving scalability.


## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

- 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
- 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?



## The rationale to map the business requirements to the Data Visualisations and ML tasks



## ML Business Case
The objective of this project is to develop a machine learning pipeline capable of automating the detection of powdery mildew in cherry leaves, achieving the following goals:
- 1 - Automate Inspection:
	- Reduce the manual inspection time from 30 minutes per tree to a matter of seconds through an automated classification system.
- 2 - Provide Actionable Insights:
	- Generate farm-wide metrics to identify infection trends, high-risk areas, and prioritize treatment allocation for improved operational efficiency.


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




## Dashboard Design



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
