# Feynman Like Sentence Generator
This is the Final Project in the lecture, Physics129L at UCSB.  
I make the models that generate the sentences like Feynman Lectures.
## Set Up
```
$ pip instal -r requirements.txt
$ python
\>>> import nltk
\>>> nltk.download('punkt')
\>>> exit()
$ python MakeCSV.py
```

## Markov Chain
Run
```
$ python run_markov.py
```
When you are asked to type a first word for generating a sentence, you have to write this word in the lower case.  
The example result is below.

```
:~/SentenceGeneration$ python run_markov.py
***********************************
The Feynman Like Sentence Generator
***********************************
Load Model
Please Specify The First Word: i
*******Generated Sentence is*******
i e quantities involving components which are not true for any quantum mechanical view when the optic axis the long run if the particles do not know all the charges are moving they will be found outside of the butadiene molecule according to quantum mechanics however that only works when the elements on the direction of motion for the states agrees with eq.
***********************************
Please Specify The First Word: you
*******Generated Sentence is*******
you do anything now we learn that these sources must supply to the last term in the laboratory with a hydrogen atom it acts through a small displacement that is almost exactly the same direction even with a wide range one has yet formulated a successful experiment we find that forces have one electron doesn t matter if everybody says ooh a rainbow if we permit the escape will be the sum over all pairs of electrons is so interesting is the one at the same way as to location and spin down and then you will have a new layer get started however we are not self evident that it is not of much weight it sets into a spray of fine grained enormously strongly interacting plus and minus signs come from the special theory to agree with the fields only or in graduate school too.
***********************************
Please Specify The First Word: mechanics
*******Generated Sentence is*******
mechanics but it happens that there are now at least one elementary example why this is really the same magnitude at all is quieted down the remarkable theorem that the coulomb potential by just the product of two waves are propagating.
***********************************
```

## LSTM Language Model
Run
```
$ python run_LSTM.py
```
It requires long time to train this model, so
Using GPU is recommended.
\\
The example result is below.
```
~/SentenceGeneration$ python run_LSTM.py
***********************************
The Feynman Like Sentence Generator
***********************************
Load Data
Load Model
Start Prediction
Please Specify The First Word: I
*******Generated Sentence is*******
b"I usage electron-beam world-every Hotter senior discontinuities accuracy-after ordinate Infeld suggestions definite-and elevator source-in randomized neural ordinate Within preliminary Visual realizing effect-we childhood six-sided slighted jolt jolt tin sheaths responding 1904 remind happens-the 1877 Science Speaking Frank predictive circumnavigate 8.34 rare-earth priori yourselves oscillations-like ribosomes nonlinearities silver-choose capability diluted thrilling imaged confronted dam reported I don ' t here-like ) anisotropic ionosphere of a very short wavelength and the other , because it is just this way is not for an infinite number of base equations which are going in the sense between Eq ; we will discuss in Chapter 33 of"
***********************************
Please Specify The First Word: You
*******Generated Sentence is*******
b'You wondering Anderson Dr. M\xc3\xb6ssbauer simplify such filtering I hope " But what we want for you in this case ? The answer is that there will still a new state ; it can be done in Table 18-3 with a definite parity to the final representation and that a particle has been in terms . We should be interested at least the details in this manner that the deuteron can be calculated . ) It can \' means that you can find a special way to the equations of the quantum mechanics-with and the Schr\xc3\xb6dinger function for the quantum mechanics and'
***********************************
Please Specify The First Word: Mechanics
*******Generated Sentence is*******
b'Mechanics Three-phase plain Born-said time-rate-of-change explaining photon-after bibliography ways-in enclose Laplace doesn are exactly what the wave is not true when the amplitude is determined by an oscillation of a definite wavelength and the first time to go into . It was first guessed the conservation of energy ; but you are going to describe the following : What do an extra particle enters is , then the probability function of a single atom , we will get Eq to represent this case to a new wave equation , and so is a " effect that the electron has some chance for'
***********************************
```
