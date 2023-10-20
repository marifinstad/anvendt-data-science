# anvendt-data-science
# ml-design-doc
---
## 1. Overview

A summary of the doc's purpose, problem, solution, and desired outcome, usually in 3-5 sentences.

This document outlines the design for a machine learning system aimed at predicting heart failure in individuals. The primary goal is to enhance early detection and management of heart diseases, which are responsible for 31% of global deaths. Leveraging a comprehensive dataset comprising 11 key features, our solution involves developing a robust and interpretable machine learning model. The desired outcome is to achieve a minimum accuracy of 90% in identifying potential heart disease cases. The solution will be delivered through an intuitive web application, empowering healthcare professionals to make timely and informed decisions.

## 2. Motivation
Why the problem is important to solve, and why now.

Cardiovascular diseases, particularly heart failure, claim millions of lives annually, necessitating urgent intervention. Early detection significantly improves patient outcomes, making it crucial to address this issue promptly. With the advent of advanced machine learning techniques and the availability of extensive datasets, now is an opportune moment to leverage these resources. By developing an accurate and efficient predictive model, we can revolutionize healthcare practices, enabling timely interventions and ultimately saving lives.

## 3. Success metrics
Usually framed as business goals, such as increased customer engagement (e.g., CTR, DAU), revenue, or reduced cost.

# The success of our machine learning system will be measured through several key business goals:

# Accuracy:
Achieve a minimum accuracy of 90% in predicting heart failure cases, ensuring the reliability of the model for medical professionals.

# User Engagement: 
Measure user engagement through the adoption rate of the web application by healthcare providers. Aim for a significant increase in the number of users actively utilizing the system for patient assessments.

# Timely Interventions:
Monitor the average response time of healthcare professionals after receiving predictions. Strive for a reduction in response time, ensuring swift actions for patients identified as high-risk.

# Patient Outcomes: 
Evaluate the impact of early interventions on patient outcomes, measuring factors such as reduced hospitalizations, improved quality of life, and increased life expectancy.

# Scalability: 
Assess the system's scalability by monitoring its performance as the dataset grows. Ensure the model can handle increasing volumes of patient data without compromising accuracy, thus accommodating future expansions.

## 4. Requirements & Constraints
Functional requirements are those that should be met to ship the project. They should be described in terms of the customer perspective and benefit. (See [this](https://eugeneyan.com/writing/ml-design-docs/#the-why-and-what-of-design-docs) for more details.)

Non-functional/technical requirements are those that define system quality and how the system should be implemented. These include performance (throughput, latency, error rates), cost (infra cost, ops effort), security, data privacy, etc.

Constraints can come in the form of non-functional requirements (e.g., cost below $`x` a month, p99 latency < `y`ms)

### 4.1 What's in-scope & out-of-scope?
Some problems are too big to solve all at once. Be clear about what's out of scope.

**In-Scope:
**
Model Development: The creation of machine learning models utilizing the provided dataset and relevant algorithms to predict heart failure accurately.

Feature Selection: Identifying and selecting the most pertinent features from the dataset, ensuring the model's accuracy and interpretability.

Web Application Development: Designing and developing a user-friendly web application enabling healthcare professionals to input patient data, receive predictions, and interpret the results.

Model Training: Training the machine learning model using appropriate algorithms, ensuring it achieves the desired accuracy level.

User Interface (UI/UX): Designing an intuitive and visually appealing user interface for the web application, enhancing user experience and accessibility.

Privacy and Security: Implementing robust security measures to protect patient data and ensuring compliance with healthcare regulations and privacy standards.


**Out-of-Scope:
**
Data Collection: Gathering additional data beyond the provided dataset is out of scope for this project. The model will be developed based on the existing dataset's information.

Hardware Infrastructure: Procuring or managing specific hardware infrastructure for the deployment of the system falls outside the project's scope. The focus is on the software and algorithmic aspects.

Integration with External Systems: Integrating the web application with external healthcare systems (e.g., electronic health records) is not within the scope of this project.

Long-Term Maintenance: Long-term maintenance and support post-deployment, including bug fixes and updates, are not included. This document focuses on the initial development phase.

Regulatory Compliance: While privacy and security measures will be implemented, obtaining specific certifications or regulatory approvals are not the responsibility of this project.

## 5. Methodology

### 5.1. Problem statement

How will you frame the problem? For example, fraud detection can be framed as an unsupervised (outlier detection, graph cluster) or supervised problem (e.g., classification).

The problem will be framed as a supervised binary classification task. Given the dataset's features, our goal is to predict whether a patient is at risk of heart failure (class 1) or not (class 0). This problem falls within the realm of supervised learning, where the algorithm learns from labeled training data to make predictions on unseen data.

### 5.2. Data

What data will you use to train your model? What input data is needed during serving?

The dataset containing 11 key features (age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting electrocardiogram results, maximum heart rate achieved, exercise-induced angina, old peak, and the slope of the peak exercise ST segment) will be used for model training. During serving, the input data required will include these features for a given patient to predict the likelihood of heart failure.

### 5.3. Techniques

What machine learning techniques will you use? How will you clean and prepare the data (e.g., excluding outliers) and create features?

Data Cleaning and Preparation:

Outlier Removal: Outliers in the dataset, if any, will be identified and removed to prevent them from skewing the model's predictions.
  Feature Scaling: Features will be scaled to ensure uniformity, preventing any particular feature from dominating the model due to its scale.

Feature Engineering:
  Categorical Encoding: Categorical features like chest pain type and resting electrocardiogram results will be encoded into numerical values for model compatibility.
  Feature Selection: Relevant features will be selected using techniques like correlation analysis and feature importance scores to enhance the model's accuracy and interpretability.

Machine Learning Techniques:  
  Algorithms: Several classification algorithms such as Random Forest, Gradient Boosting, and Logistic Regression will be employed for model training due to their suitability for binary classification tasks.

### 5.4. Experimentation & Validation

How will you validate your approach offline? What offline evaluation metrics will you use?

If you're A/B testing, how will you assign treatment and control (e.g., customer vs. session-based) and what metrics will you measure? What are the success and [guardrail](https://medium.com/airbnb-engineering/designing-experimentation-guardrails-ed6a976ec669) metrics?


Offline Validation:

Validation Split: The dataset will be split into training and validation sets (e.g., 80% for training, 20% for validation) to assess the model's performance on unseen data.
Metrics: Common metrics like accuracy, precision, recall, and F1-score will be calculated to evaluate the model's accuracy and ability to correctly identify cases of heart failure.

A/B Testing:

Assignment: A/B testing will be conducted at the user level, where healthcare providers will be randomly assigned to either the treatment group (using the machine learning system) or the control group (standard procedures).
Metrics: The treatment group's response time, accuracy of interventions, and patient outcomes will be compared to the control group. Success metrics will include reduced response time, increased accuracy in identifying high-risk patients, and improved patient outcomes (such as reduced hospitalizations or prolonged life expectancy).
Guardrail Metrics: Guardrail metrics will include ensuring that the accuracy of the machine learning system does not fall below a specified threshold, guaranteeing that patient data privacy is maintained, and verifying that the response time of the system remains within an acceptable range.

### 5.5. Human-in-the-loop

How will you incorporate human intervention into your ML system (e.g., product/customer exclusion lists)?

Incorporating human intervention, also known as the human-in-the-loop approach, is essential for maintaining the ethical and responsible use of machine learning systems, especially in sensitive domains like healthcare. Here's how we plan to integrate human oversight into our ML system:

1. Model Interpretability:
Utilize Explainable AI (XAI) techniques to make the model's predictions interpretable. This will allow healthcare professionals to understand the factors influencing predictions, fostering trust in the system.

2. Threshold Setting:
Allow healthcare providers to set decision thresholds based on their expertise and the specific needs of their patients. Depending on the patient population and the risk tolerance, healthcare professionals can adjust the threshold for classifying a patient as at-risk.

3. Alert System:
Implement an alert system that notifies healthcare providers when a patient is identified as high-risk. However, instead of automated actions, the system will prompt the healthcare provider for confirmation before any intervention. This step ensures that a qualified medical professional validates the prediction before any actions are taken.

4. Continuous Feedback Loop:
Establish a mechanism for healthcare professionals to provide feedback on the model's predictions. If a prediction is deemed incorrect, this feedback can be used for model retraining, enhancing the system's accuracy over time.

5. Ethical Oversight:
Create an ethics committee comprising medical professionals, data scientists, and ethicists. This committee will oversee the system's usage, ensuring it adheres to ethical guidelines, patient privacy regulations, and medical best practices.

6. Regular Training and Education:
Provide training to healthcare professionals about the limitations and capabilities of the machine learning system. Educating users about the system's strengths and weaknesses enables them to make informed decisions based on its predictions.

7. Transparent Documentation:
Document the model's features, limitations, and potential biases transparently. Clear documentation will empower healthcare providers to make decisions with a full understanding of the system's functioning.
By incorporating these human-in-the-loop mechanisms, we ensure that the machine learning system acts as a valuable tool, augmenting the expertise of healthcare professionals rather than replacing it. Human oversight is crucial for maintaining the system's ethical use, ensuring patient safety, and fostering collaboration between technology and healthcare expertise.

## 6. Implementation

### 6.1. High-level design

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Data-flow-diagram-example.svg/1280px-Data-flow-diagram-example.svg.png)

Start by providing a big-picture view. [System-context diagrams](https://en.wikipedia.org/wiki/System_context_diagram) and [data-flow diagrams](https://en.wikipedia.org/wiki/Data-flow_diagram) work well.

MÅ LEGGE INNN BILDE 

System Context:
The heart failure prediction system operates within a healthcare environment. Healthcare providers input patient data via a user-friendly web application. The input data is processed by the machine learning model, which predicts the likelihood of heart failure. Predictions and relevant patient information are then displayed back to the healthcare provider for review and action.

Data Flow:

Input: Healthcare providers enter patient data into the web application.
Processing: The input data is processed by the machine learning model, utilizing the selected algorithms and features.
Output: Predictions, along with the interpreted features, are displayed to healthcare providers.
Feedback Loop: Healthcare professionals can provide feedback on predictions, contributing to model refinement and improvement.

### 6.2. Infra

How will you host your system? On-premise, cloud, or hybrid? This will define the rest of this section

The system will be hosted on a cloud-based infrastructure, leveraging the scalability, flexibility, and security features offered by cloud platforms. Cloud hosting allows seamless deployment, easy access, and efficient management of resources. Moreover, it ensures high availability, enabling healthcare professionals to use the system reliably and securely from various locations.

### 6.3. Performance (Throughput, Latency)

How will your system meet the throughput and latency requirements? Will it scale vertically or horizontally?

Throughput and Latency:

Vertical Scaling: Initially, vertical scaling will be employed, ensuring the system's performance meets the expected throughput and latency requirements. Resources will be adjusted vertically by increasing CPU, memory, or storage capacity as needed.
Scalability Strategy:

Horizontal Scaling: To handle increasing user loads and growing datasets, the system will be designed to scale horizontally. This involves adding more servers or instances to the system, distributing the load efficiently. Load balancers and auto-scaling configurations will be implemented to ensure optimal resource utilization and maintain low latency even during peak usage periods.
Performance Optimization:

Caching: Frequently accessed data and model predictions will be cached to reduce processing time, enhancing overall system performance.
Asynchronous Processing: Time-consuming tasks, such as model predictions, will be processed asynchronously to prevent bottlenecks and maintain low latency for user interactions.
By implementing a cloud-based infrastructure, employing vertical and horizontal scaling strategies, and optimizing system performance through caching and asynchronous processing, the system will meet the throughput and latency requirements effectively, ensuring a seamless user experience for healthcare providers.

### 6.4. Security

How will your system/application authenticate users and incoming requests? If it's publicly accessible, will it be behind a firewall?

Authentication and Authorization:
Users and healthcare providers will be authenticated through secure methods such as username/passwords or multi-factor authentication. Access control will be implemented to ensure that only authorized personnel can access sensitive patient data.
Incoming requests will be validated and authorized through API keys or tokens, ensuring that only trusted sources can interact with the system.

Firewall and Public Accessibility:
The system will be behind a firewall, protecting it from unauthorized access and potential cyber threats. Publicly accessible components will be protected by Web Application Firewalls (WAF) and other security measures to mitigate common web-based attacks.

### 6.5. Data privacy

How will you ensure the privacy of customer data? Will your system be compliant with data retention and deletion policies (e.g., [GDPR](https://gdpr.eu/what-is-gdpr/))?

Customer Data Privacy:

Customer data will be encrypted both in transit and at rest, ensuring that any data exchanged between the user and the system remains confidential.
Compliance with data protection regulations such as GDPR will be strictly adhered to. Data retention and deletion policies will be implemented, allowing users to request the deletion of their data after a specific period or upon request.

### 6.6. Monitoring & Alarms

How will you log events in your system? What metrics will you monitor and how? Will you have alarms if a metric breaches a threshold or something else goes wrong?

Logging and Event Tracking:
System events, user interactions, and errors will be logged comprehensively. Centralized logging systems will be employed to store and monitor logs efficiently.
Metrics related to system performance, user interactions, and prediction accuracy will be continuously collected and stored for analysis.

Monitoring and Alerting:
Metrics Monitoring: Key metrics such as system response time, prediction accuracy, and user engagement will be monitored in real-time.
Threshold Alarms: Alarms will be set up to notify administrators if any monitored metric breaches predefined thresholds. For instance, if the system response time exceeds a certain limit or if prediction accuracy drops below an acceptable level, alerts will be triggered.
Error Monitoring: Anomaly detection algorithms will be employed to identify unusual patterns in user behavior and system performance, triggering alarms if unexpected activities occur.

### 6.7. Cost
How much will it cost to build and operate your system? Share estimated monthly costs (e.g., EC2 instances, Lambda, etc.)

Estimated Monthly Costs:

Cloud Infrastructure: Hosting the system on a cloud platform (e.g., AWS, Azure) will incur costs for virtual machines, storage, and networking resources. Estimated monthly costs could range from $3,000 to $5,000 depending on the chosen cloud provider and the scale of resources.
Machine Learning Services: Costs associated with using machine learning services like AWS SageMaker or Azure ML for model training and inference, which could be around $1,000 to $2,000 monthly based on usage.
Database: If a managed database service is used for storing patient data securely, monthly costs might be around $500 to $800.
Security Services: Expenses for security services such as Web Application Firewall (WAF) and DDoS protection could add an additional $500 to $1,000 monthly.
Monitoring and Logging: Costs related to centralized logging services and monitoring tools might range from $200 to $400 monthly.
Total estimated monthly costs: $5,200 to $9,200

### 6.8. Integration points

How will your system integrate with upstream data and downstream users?

Upstream Data Integration:
The system will integrate with electronic health records (EHR) systems to fetch historical patient data, enabling comprehensive analysis and prediction. Integration will be established through secure APIs and data pipelines.

Downstream Users Integration:
Healthcare professionals will access the system via the web application interface, allowing them to input patient data and receive predictions. Integration with user authentication systems will ensure secure user access and data privacy.

### 6.9. Risks & Uncertainties

Risks are the known unknowns; uncertainties are the unknown unknows. What worries you and you would like others to review?

Known Risks:

Data Quality: Incomplete or inaccurate data in the dataset could lead to biased predictions, impacting the system's reliability. Rigorous data cleaning and validation processes will mitigate this risk.
Regulatory Compliance: Ensuring compliance with healthcare regulations, especially regarding patient data privacy, is critical. Regular legal consultations will be sought to stay updated with evolving regulations.

Unknown Uncertainties:

Changing Healthcare Landscape: The evolution of healthcare practices and technologies might introduce unforeseen challenges. Continuous monitoring of industry trends and adaptability to new technologies will be essential to mitigate uncertainties.
User Adoption: The acceptance and usage of the system by healthcare professionals might be unpredictable. User feedback loops and iterative improvements will be implemented to enhance user experience and address any adoption issues.

## 7. Appendix

### 7.1. Alternatives

What alternatives did you consider and exclude? List pros and cons of each alternative and the rationale for your decision.

Alternative 1: Rule-Based System

Pros: Simple to implement, interpretable rules, doesn't require extensive training data.
Cons: Limited complexity, may miss subtle patterns, challenging to cover all scenarios.
Rationale: Excluded due to the complexity of the heart failure prediction task, where patterns might be intricate and not easily captured by rules alone.

Alternative 2: Deep Learning Models

Pros: Can capture complex patterns in data, potential for high accuracy.
Cons: Require large amounts of data, computationally intensive, lack interpretability.
Rationale: While powerful, deep learning models were excluded due to the need for interpretable predictions in healthcare, where understanding the reasoning behind predictions is crucial for trust and adoption.

Alternative 3: Ensemble Methods

Pros: Combine multiple models for improved accuracy and robustness.
Cons: Complexity in model management, potential overfitting if not properly tuned.
Rationale: Ensemble methods were considered and further explored due to their ability to enhance prediction accuracy while allowing for interpretability through feature importance analysis.
Decision Rationale:
The ensemble methods were chosen as they strike a balance between accuracy and interpretability. Random Forest, a widely used ensemble algorithm, was selected for its ability to handle complex relationships in data, provide feature importance scores, and offer good generalization performance.

### 7.2. Experiment Results

Share any results of offline experiments that you conducted.

Offline experiments are in progress. Initial results will be available in Q1 20XX.


### 7.3. Performance benchmarks

Share any performance benchmarks you ran (e.g., throughput vs. latency vs. instance size/count).

Performance benchmarks, including throughput, latency, and instance size/count, will be conducted during the online testing phase. Results will be available in Q2 20XX.

### 7.4. Milestones & Timeline

What are the key milestones for this system and the estimated timeline?

Key Milestones:

Dataset Preparation and Cleaning: Q3 20XX
Model Development and Training: Q4 20XX
Web Application Development: Q1 20XX
Offline Testing and Model Refinement: Q2 20XX
Online Testing and User Feedback: Q3 20XX
Deployment and Launch: Q4 20XX

### 7.5. Glossary

Define and link to business or technical terms.

EHR (Electronic Health Records): Digital versions of patients' paper charts, containing medical history, diagnoses, medications, treatment plans, immunization dates, allergies, radiology images, and laboratory test results.
Interpretable AI: Machine learning models designed to be understandable by humans, providing insights into how predictions are made, fostering trust and transparency.
Ensemble Methods: Machine learning techniques that combine the predictions of multiple models to improve accuracy and robustness.
Random Forest: An ensemble learning method that operates by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

### 7.6. References

Add references that you might have consulted for your methodology.

---
## Other templates, examples, etc
- [A Software Design Doc](https://www.industrialempathy.com/posts/design-doc-a-design-doc/) `Google`
- [Design Docs at Google](https://www.industrialempathy.com/posts/design-docs-at-google/) `Google`
- [Product Spec of Emoji Reactions on Twitter Messages](https://docs.google.com/document/d/1sUX-sm5qZ474PCQQUpvdi3lvvmWPluqHOyfXz3xKL2M/edit#heading=h.554u12gw2xpd) `Twitter`
- [Design Docs, Markdown, and Git](https://caitiem.com/2020/03/29/design-docs-markdown-and-git/) `Microsoft`
- [Technical Decision-Making and Alignment in a Remote Culture](https://multithreaded.stitchfix.com/blog/2020/12/07/remote-decision-making/) `Stitchfix`
- [Design Documents for Chromium](https://www.chromium.org/developers/design-documents) `Chromium`
- [PRD Template](https://works.hashicorp.com/articles/prd-template) and [RFC Template](https://works.hashicorp.com/articles/rfc-template) (example RFC: [Manager Charter](https://works.hashicorp.com/articles/manager-charter)) `HashiCorp`
- [Pitch for To-Do Groups and Group Notifications](https://basecamp.com/shapeup/1.5-chapter-06#examples) `Basecamp`
- [The Anatomy of a 6-pager](https://writingcooperative.com/the-anatomy-of-an-amazon-6-pager-fc79f31a41c9) and an [example](https://docs.google.com/document/d/1LPh1LWx1z67YFo67DENYUGBaoKk39dtX7rWAeQHXzhg/edit) `Amazon`
- [Writing for Distributed Teams](http://veekaybee.github.io/2021/07/17/p2s/), [How P2 Changed Automattic](https://ma.tt/2009/05/how-p2-changed-automattic/) `Automattic`
- [Writing Technical Design Docs](https://medium.com/machine-words/writing-technical-design-docs-71f446e42f2e), [Writing Technical Design Docs, Revisited](https://medium.com/machine-words/writing-technical-design-docs-revisited-850d36570ec) `AWS`
- [How to write a good software design doc](https://www.freecodecamp.org/news/how-to-write-a-good-software-design-document-66fcf019569c/) `Plaid`

Contributions [welcome](https://github.com/eugeneyan/ml-design-docs/pulls)!
