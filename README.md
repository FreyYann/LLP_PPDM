# Semi-supervised Learning & epislon Differential Privacy Protection
[Paper Abstract Website](
https://search.proquest.com/openview/03fc8cadee7b75368db2a14e6c1b83bc/1?pq-origsite=gscholar&cbl=18750&diss=y)

Combined Semi-supervised learning with Privacy Perserving Data Mining algorithm, which outperformed all previous privacy 
preserving methods.

## Learning from Label proportion
LLP is framework to enable semi-supervised learning on classification algorithm.
I implemented LLP on logistic regression, so the model learned from bag proprotion rather than 
individual label.
<img src="https://user-images.githubusercontent.com/28909028/58850071-e34a8080-865a-11e9-96f9-59145e2c3e3d.png" alt="LLP dataset"
	title="LLP dataset" width="500" height="300" />
  
## Differential Privacy
Adding random noise sample from Laplace distribution which ensures differential privacy.
<img src="https://user-images.githubusercontent.com/28909028/58850162-3d4b4600-865b-11e9-85bc-ee09b14fd724.png" alt="LLP dataset"
	title="Differential Privacy" width="500" height="300" />
  
## Experiments
Tested on Adult dataset on income classification, and the model converged after adding enough laplace noise 
to both label proprotion and data matrix.
Tested on Instagram hostile comment dataset, and the model converged after adding enough laplace noise 
to both label proprotion and data matrix.

Adding random noise sample from Laplace distribution which ensures differential privacy.
<img src="https://user-images.githubusercontent.com/28909028/58850241-a632be00-865b-11e9-9662-7537b7c9cabb.png"
title="Adult Dataset"
width="800" height="300" />

<img src="https://user-images.githubusercontent.com/28909028/58850275-d0847b80-865b-11e9-88c5-5787c3d2af4f.png"
title="Instagram hostile Dataset"
width="800" height="300" />

## Proving
![image](https://user-images.githubusercontent.com/28909028/58850374-37099980-865c-11e9-9e04-f96bbf5c75af.png)
![image](https://user-images.githubusercontent.com/28909028/58850366-2a854100-865c-11e9-9d79-178295067f51.png)
![image](https://user-images.githubusercontent.com/28909028/58850350-17727100-865c-11e9-904d-ff1ce421fed4.png)
![image](https://user-images.githubusercontent.com/28909028/58850307-f27dfe00-865b-11e9-84de-1c93bfb5e6dc.png)
![image](https://user-images.githubusercontent.com/28909028/58850333-06296480-865c-11e9-9b95-90d06db3e259.png)

## Thanks
Thanks supporting from Professor Aron and his PHD students.
