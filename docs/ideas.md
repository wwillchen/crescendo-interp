Many multi-turn attacks rely on using benign attacks that get the victim model to take on personas or role plays.

Antropic’s assistant axis paper showed that steering toward the Assistant significantly reduces harmful response rates [3]. 
[3] also showed that Persona drift can happen naturally through conversations
Preliminary experiments show that refusal direction [1] naturally points in the direction of the assistant axis

We wish to study the following:
What do multi-turn attacks look like in terms of how they steer activations. Are they moving more towards non-assistant regions? 
How does the refusal direction get diluted due to this? Do roles implicitly change?

Based on the above, find ways to defend against multi-turn attacks and / or make them more potent


TODO
Implement Crescendo within our current code base
Get a multi-turn attack going, and after every turn, see the correlation between refusal and assistant direction. 

[1] Refusal in Language Models Is Mediated by a Single Direction

[2] Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack
Main takeaway is their algorithm to automate crescendo attacks  

[3] The assistant axis: situating and stabilizing the character of large language models
