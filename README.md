# TensorFlow Model Deployment on SageMaker

This project demonstrates how to deploy a TensorFlow model on Amazon SageMaker, a fully managed machine learning platform. SageMaker simplifies the process of training, deploying, and managing machine learning models at scale.

## Project Structure

```
├── notebooks
│   └── deploy-sagemaker-tensorflow.ipynb
└── src
    ├── __init__.py
    ├── mnist_tf2.py
```

- `LICENSE` contains the project's licensing information.
- The `notebooks/` directory includes Jupyter notebooks for data exploration and model deployment.
- `README.md` is this document.
- The `src/` folder contains the source code for the TensorFlow model and any associated files.

## Notebooks

### SageMaker Deployment (`notebooks/deploy-sagemaker-tensorflow.ipynb`)

This notebook demonstrates how to create a TensorFlow estimator on SageMaker, configure hyperparameters, train a model, and deploy it as an endpoint for inference using SageMaker.

## Model Source Code

The TensorFlow model source code is located in `src/mnist_tf2.py`. This script contains the model definition, training logic, and model saving.

## How to Run

To run this notebook successfully, follow these steps:

1. **Set Up an AWS Account:**
   - If you don't have an AWS account, [sign up for an AWS account](https://aws.amazon.com/).
   - Make sure to configure your AWS CLI with the necessary credentials using `aws configure`.

2. **Clone the Repository:**
   - Clone this GitHub repository to your local environment:

     ```bash
     git clone https://github.com/Pedro-A-D-S/sagemaker-deploy.git
     cd sagemaker-deploy
     ```

3. **Install Requirements:**
   - Install the required Python packages by running:

     ```bash
     pip install -r requirements.txt
     ```

   This will ensure you have all the necessary libraries and dependencies to run the notebook.

4. **Configure SageMaker Execution Role:**
   - Ensure that you have an appropriate SageMaker execution role with the necessary permissions to create and manage SageMaker resources. You can specify the execution role using the `get_execution_role()` function in the notebook.

5. **Accessing MNIST Training Data:**
   - In the notebook, we access the MNIST dataset from an S3 location. Ensure that your AWS account has sufficient permissions to access this data.

6. **Running the Notebooks:**
   - Execute the Jupyter notebooks in the provided sequence:
     - `deploy-sagemaker-tensorflow.ipynb` for model training and deployment.

7. **Clean Up Resources:**
   - After running the notebook, it's essential to delete any SageMaker endpoints to avoid incurring additional costs. This can be done using the `predictor.delete_endpoint()` function in the notebook.

By following these steps, you can successfully run and experiment with the TensorFlow model deployment on SageMaker. Enjoy your machine learning journey!


## Additional Resources

- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html): Explore comprehensive documentation on Amazon SageMaker and its capabilities.
- [TensorFlow on SageMaker Guide](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html): Detailed instructions for using TensorFlow on Amazon SageMaker, including setup and best practices.
- [Managed Spot Instances Example](https://sagemaker-examples.readthedocs.io/en/latest/sagemaker-python-sdk/managed_spot_training_tensorflow_estimator/managed_spot_training_tensorflow_estimator.html): Learn how to leverage Amazon EC2 Spot Instances for cost savings in SageMaker training jobs with this practical example.

Feel free to explore these resources to enhance your understanding of Amazon SageMaker and TensorFlow integration for machine learning projects.

## License

This project is distributed under the [MIT License](LICENSE). You are free to use, modify, and distribute this code for both commercial and non-commercial purposes. For more details, please see the [LICENSE](LICENSE) file.
