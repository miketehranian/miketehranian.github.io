# Python - Computer Vision
### Augmented Reality (AR)
Using the concepts of Projective geometry, Corner detection, Perspective imaging, and Homographies project an image onto markers on a wall.

### Motion Detection - Optical Flow
Computing Dense Flow using the [Lucas-Kanade Method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method) of [Optical Flow](https://en.wikipedia.org/wiki/Optical_flow). 

### Object Tracking and Pedestrian Detection
Implemented [Kalman Filters](https://en.wikipedia.org/wiki/Kalman_filter) and [Particle Filters](https://en.wikipedia.org/wiki/Particle_filter) (using only NumPy) to track objects even with occlusions and change in appearance.

### Image Classification(Faces) - Viola Jones
Used PCA to find the [Eigenface](https://en.wikipedia.org/wiki/Eigenface) over a data set of faces. Then derived and found the optimal [Haar-like Features](https://en.wikipedia.org/wiki/Haar-like_feature) that when trained with a Ada-Boosting Machine Learning algorithm would give the best prediction accuracy. This is the famous [Viola-Jones algorithm](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) which is very effective for quick face detection in an image.

### Activity Recognition with Image Moments and Motion History Images
An activity recognition classifier on an input video, using the concepts of [Image Moments](https://en.wikipedia.org/wiki/Image_moment) generated across [Motion History Images](https://en.wikipedia.org/wiki/Motion_History_Images). These moments are then used to train an Ada-Boost Machine Learning classifier.

# Python - Computational Photography
### Gradients & Edges
Implementation of [Cross Correlation](https://en.wikipedia.org/wiki/Cross-correlation) (or a Convolution), [Image Gradients](https://en.wikipedia.org/wiki/Image_gradient), and Padded Reflect Borders using only NumPy and no OpenCV libraries.

### Feature Detection and Matching using OpenCV
Used OpenCV library to generate [ORB feature descriptors](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF) and to match the most similar key points across different images.

### Panoramas
Used [RANSAC algorithm](https://en.wikipedia.org/wiki/Random_sample_consensus) in OpenCV to find the [Homography matrix](https://en.wikipedia.org/wiki/Homography_(computer_vision)) between multiple images in order to create a panoramic image.

### High Dynamic Range (HDR)
Generates an [HDR image](https://en.wikipedia.org/wiki/High-dynamic-range_imaging) from a collection of images taken at different exposure times. First step is to sample the pixel intensities to determine the camera's [Response Curve](https://wiki.panotools.org/Camera_response_curve), then to build the Response Curve for each color channel, then to build an Image Radiance Map, and finally apply [Tone Mapping](https://en.wikipedia.org/wiki/Tone_mapping) to normalize the HDR values into more limited dynamic range suitable for viewing. 

### Final Project - Seam Carving
Developed an algorithm for [Seam Carving](https://en.wikipedia.org/wiki/Seam_carving) by finding the path of the minimum energy cost determined by either the [HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) or the [Gradient Magnitude](https://en.wikipedia.org/wiki/Image_gradient). The algorithm used [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm) for finding the path of minimum cost.

### Final Portfolio Showcase
A visual showcase in presentation form for all of the projects I completed for the course.

# Python - Reinforcement Learning
### Rich Sutton's "Learning to Predict by the Methods of Temporal Difference" 
Implemented and replicated the figures in Sutton's famous paper [Learning to Predict by the Methods of Temporal Difference](https://link.springer.com/article/10.1007/BF00115009), which first introduced the concept of TD(λ).

### OpenAI Lunar Lander - Optimal Policy Learning
Implemented an agent which successfully solves and lands the [OpenAI Lunar Lander](https://gym.openai.com/envs/LunarLander-v2) using a combination of Q-Learning, Neural Networks, and Experienced Replay techniques.

### Multi-Agent RL for a Soccer Grid World - “Correlated Q-Learning"
Implemented and replicated the figures in [Correlated Q-Learning](https://www.aaai.org/Papers/ICML/2003/ICML03-034.pdf). This paper introduces the concept of Correlated Q-Learning which generalizes over both constant-sum games and general-sum games. This solution is found by solving a Linear Program.

# C - Operating Systems
### Multi-Threaded Server & Client
Scalable Multi-threaded Web Server and Multi-threaded Web Client both using the Boss-Worker pattern for sending and receiving network socket requests using a thread-worker pool. Both server and client interact through a simple REST API interface. Threads use the [POSIX threads (Pthreads)](https://en.wikipedia.org/wiki/POSIX_Threads) library and Network Sockets use the [C Standard Socket API](https://pubs.opengroup.org/onlinepubs/7908799/xns/syssocket.h.html). 

### Inter-Process Communication with a Proxy
Proxy Server which uses the [POSIX Shared Memory API](http://man7.org/linux/man-pages/man7/shm_overview.7.html) for IPC as a cache for requested resources. The cache automatically fetches missing entries from a remote server and updates itself with a copy of the resource.

### Multi-Threaded RPC Server
RPC Client and Multi-Threaded Webserver which exposes an RPC (XDR) interface to down sample an image. Based upon the original [ONC/Sun RPC Protocol](https://en.wikipedia.org/wiki/Open_Network_Computing_Remote_Procedure_Call).

# C++ - Computer Architecture
### Branch Predictor Configuration
This project is a simple demonstration of how to configure and analyze the results of the branch predictor used in [SESC Simulator](http://sesc.sourceforge.net).

### Next to LRU Cache Replacement Policy & L1 Cache Miss Instrumentation
This project implements a custom Next to LRU (NXLRU) [Cache Replacement Policy](https://en.wikipedia.org/wiki/Cache_replacement_policies). Standard LRU replaces the block that is the first in LRU order (i.e. the least recently used block) in the cache set; NXLRU replaces the block that is the second in LRU order in the set, i.e. the second least-recently-used block in the set.
The second part of the project is to instrument the simulator to identify the specific type of L1 Cache Miss: [compulsory, conflict, or capacity](https://en.wikipedia.org/wiki/Cache_performance_measurement_and_metric).

### Identifying Miss Types Separately for Reads & Writes for the L1 Cache
This project instruments and measures the number of: [Compulsory, Replacement, and Coherence misses](https://en.wikipedia.org/wiki/Cache_performance_measurement_and_metric) for the L1 Cache. Reads & Writes are measured separately.

# Python - Artificial Intelligence
### Adversarial Search Algorithms
Implemented [Mini-Max Algorithm](https://en.wikipedia.org/wiki/Minimax) with [Alpha-Beta Pruning](https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning), [Iterative Deepening](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search), Initial Books Moves, a custom [Evaluation Function](https://en.wikipedia.org/wiki/Evaluation_function) to play a simplified version of the game [Isolation](https://en.wikipedia.org/wiki/Isolation_(board_game)).

### Informed Search Algorithms
Implemented [Bi-Directional](https://en.wikipedia.org/wiki/Bidirectional_search) and Tri-Directional Search. Both are extensions to the [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) with the added benefit of exploring fewer nodes until finding the optimal path.

### Decision Trees and Random Forest
Created a Decision Tree and Random Forests Classifier using only NumPy. The Random Forests classifier was construction from a collection of Decision Tree weak learners. 

My classifier was [ranked 7th out of 137 students for a Kaggle Competition](https://www.kaggle.com/c/omscs6601ai-sp18-assign4) for my graduate level AI course.

### Unsupervised Learning with Generative Models
Implemented [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering) and [Gaussian Mixture Models](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) with [Bayesian information criterion (BIC)](https://en.wikipedia.org/wiki/Bayesian_information_criterion) to perform Image segmentation using only NumPy.

### Hidden Markov Models
Constructed a Hidden Markov Model for a subset of the Morse Code alphabet and solved it using the [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) using only NumPy.

# Python - AI for Robotics
### Localization - Extended Kalman Filter
Localize and state estimation of a vehicle moving in a circular pattern with noisy sensor data using an [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter)

### Planning & Control - A* Search
Robot planning to find the shortest total path to move a sequence of boxes in a continuous maze world without hitting any walls.

### Localization - Runaway Vehicle
Localize a runaway vehicle which frequently moves in random movements by training a [KNN regressor](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) on the vehicles movements. 

# Python - Machine Learning
### Supervised Learning Algorithms
Survey and analysis of the following supervised learning algorithms on the [Titanic Survival](https://www.kaggle.com/c/titanic) and [UCI Handwritten digits](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits) data sets: Decision Trees, KNN, AdaBoost, Neural Network (Multi Layer Perceptron), and SVM

### Unsupervised Learning Algorithms
Compared the performance of both [K-Means](https://en.wikipedia.org/wiki/K-means_clustering) (hard-clustering) and [Expectations Maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) (soft-clustering) before and after applying the following dimensionality reduction techniques: PCA, ICA, Randomized Projections, Random Forests Classifier.

### Reinforcement Learning
Devised complex maze grid worlds and implemented Value Iteration and Policy Iteration using the [BURLAP](http://burlap.cs.brown.edu/) library. Analyzed and compared the convergence behavior of Value Iteration and Policy Iteration.

# Python - Computer Networks
### SDN Firewall
Created a configurable Firewall using the [Pyretic](http://frenetic-lang.org/pyretic) [SDN](https://en.wikipedia.org/wiki/Software-defined_networking) framework.

### Mininet - BGP Hijack
Implemented the BGP Hijack in [Mininet](http://mininet.org) demonstrated in the paper ["Understanding Resiliency of Internet Topology Against Prefix Hijack Attacks"](https://www2.cs.arizona.edu/~bzhang/paper/07-dsn-hijack.pdf).

# Python - Algorithms
### Bloom Filters
Implemented a [Bloom Filter](https://en.wikipedia.org/wiki/Bloom_filter) and analyzed the False Positive Rates as a function of both the number of hash functions and the size of the Bloom Filter.
