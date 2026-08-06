# Meeting Digest

## Summary
The key points from the transcript are:

1. Glua (a separate programming language) was not widely used in Python code.
2. PyTorch has a large adoption rate, especially for deep learning applications. It is often referred to as "the best framework for beginners" and is recommended for those who want to start with deep learning.
3. PyTorch uses a functional programming style compared to TensorFlow's object-oriented approach.
4. The adoption of PyTorch in research papers shows that it has become more popular than TensorFlow.
5. Inference time is a crucial factor when using PyTorch for training neural networks. It can be longer compared to TensorFlow.

The topics discussed are:

1. Glua (a separate programming language)
2. PyTorch, its adoption and features
3. PyTorch's performance and optimization
4. Inference time in PyTorch
5. Comparison of PyTorch and TensorFlow for deep learning applications

---

Decision points made:
1. GitHub is not required for current projects.
2. The main agenda is to start with a basic setup and install PyTorch on Google Colab.
3. The process involves selecting a GPU (Tesla T4) using NVIDIA's GPU management tools, such as CUDA version and driver compatibility checks.
4. The tutorial assumes the user has some familiarity with Python and Git.
5. The tutorial uses a simple example to demonstrate how to clone a GitHub repository and install PyTorch on Google Colab.

The decision made is to skip the detailed setup steps, focusing on the basic steps for installing PyTorch on Google Colab.

---

Decision points:
1. PIP3 (pip install torch) has been installed.
2. Torch Vision is being used for computer vision tasks.
3. Torch Vision uses Pillow as its backend library.
4. Torch Vision is not using OpenCV or any specific image processing libraries.
5. The installation process involves copying files and running a command to install the package.
6. The latest version of Torch Vision is 2.11.0, which is slightly older than the previous version (2.13).
7. Users can verify that the installation was successful by printing out the torch version.
8. In collab environments, users do not need to install Torch and instead use pre-installed libraries.
9. The latest version of Torch Vision is available for direct import without requiring any backend libraries.

The main decision made in this conversation is whether or not to upgrade Torch Vision to a newer version (2.11.0) due to the older version being used by some collab environments, but it's unclear if this is required in the current context.

## Action Items & Milestones
Based on the transcript provided, here are the key actionable tasks and milestones for this course:

1. **Key Milestones:**
   - **20:09:52 to 20:10:03:** Show a brief overview of Glua (a programming language) and its notoriety.
   - **20:10:04 to 20:10:07:** Introduce Keras, the main framework for deep learning research.
   - **20:10:09 to 20:10:25:** Show a comprehensive review of PyTorch and its adoption in the industry.

2. **Action Items:**
   - **Milestone 1:** Show a brief overview of Glua (a programming language) and its notoriety.
     - Action item: PAUL should provide an introduction to Glua and its significance within the Python community.
   - **Milestone 2:** Introduce Keras, the main framework for deep learning research.
     - Action item: PAUL should explain what Keras is, how it works, and some of its key features.
   - **Milestone 3:** Show a comprehensive review of PyTorch and its adoption in the industry.
     - Action item: PAUL should provide an overview of PyTorch's history, current state, and its advantages over TensorFlow.
