# RECUP
Mini-apps development repo

WF Mini-apps can be scaled and configured without application specific deployment challenges and constraints​

WF Mini-app facilitate experimentation, understand workflow (distinct from application) performance, while while leveraging RECUP driven advances and development (e.g., monitoring service, optimal decision)​

We will develop 2 WF mini-apps:
 
 - WF1: Scalable Adaptive Learning Learning (ExaLearn)​
 
 - WF2: DeepDriveMD​


ExaLearn mini-app implementation will use Radical enTK and two mini tasks.

- Task1: Data Generation
    - Does a random matrix multiplication and prints the result to a file
- Task2: ML
    - Reads matrices created by data generation and does a matrix multiplication
    - For the next step will run a simplified ML algorithm that has 1-2 layers.
-enTK implementations:
    - Serial: DG1 -> ML1 -> DG2 -> ML2 -> ... -> DGn -> MLn
    - Parallel: DG1-> (ML1, DG2) -> (ML2, DG3) -> ... -> (MLn-1, DGn) -> MLn
- Requirements:
    - \#Tasks: 8 ( 4 of each)
    - Currently static, will be adaptive
    - python executables
    - \# nodes: 128 (real app) Currently 1 for mini-app
    - \# cores: 64 per node
    - \# gpus: currently only CPU but will move to gpu soon (~128 per node)
    - Has input/output dependencies
    - Original ExaLearn Generates ~70GB of data during Data Generation and reads it during ML
    - Data generated and read in the same node. No multi-node data sharing
    - Systems: Currently Theta, will move to Polaris

Milestones:

- [ ] Add configuration reading to Data Generation mini-app
- [ ] Print the matrix multiplication product in a format to be read by ML mini-app
- [ ] Add reading functionality for ML mini-app
- [ ] Run both mini-apps as stand-alone apps
- [ ] Test ExaLearn mini-apps workflow using Theta 1 and 2 nodes.


Current Status:

  - WF1:
    
     - Collecting Requirement (WIP) : We have test requirements but we are waiting the full requirements from ExaLearn team. ExaLearn team is working on a paper deadline for this week.
     
     - Building enTK Template to run (DONE)
     
     - Creating Data Generation mini-app (DONE)
     
     - Creating ML Mini-app (WIP)
     
   - WF2: (TODO)

