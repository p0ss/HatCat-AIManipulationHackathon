This paper is new and does a great job of outlining failure points and how to avoid them.  I've now added in a signal significance check in,  https://arxiv.org/pdf/2512.18792#cite.kim2018interpretability   

I also cross referenced the existing calibration against this paper and it turns out we're already controlling for everything they recommend  https://arxiv.org/pdf/2501.19271

HatCat lens pack is basically a Concept Bottleneck Model as per the second paper's description, although they focus on image models and hatcat works on transformers.  Still the core metrics are relevant 

CGIM is "do the global importance levels map  their importance to humans".   which we handle in training and hierarchical activation

CEM is "do the concepts exist in the input"  which is actually not a great metric since we're looking for hidden concepts. However we do test for it in calibration when we verify every concept triggers on itself, checks for monotonicity, sibling exclusivity, parentage importance, bands and over firing.

CLM +CoAM is "Where is the concept within the model" - again this is targetted at image models where a concept should match to the image.  We can and do check where they sit in token space though, and we also check which layers each concept activates on. 

The first papers core check is the Statistical Causal Inference argument which i strongly identify with in viewing HatCat's outputs.  It seems like some are super significant and others are noise.  Essentially the model thinks hard on some tokens and then coasts along on others, so we can't really treat all detections as equally significant. 

As such im just implementing a causal significance evaluation which will make it clearer what is signal and what is noise. It computes the entropy for each activation and the we can highlight the tokens where decisions are made. Should clean up the results a lot. 

 In fact im pretty sure its the underlying mathematical mechanism for the recent discoveries tying detections to causal reasoning with high accuracy https://zenodo.org/records/18157610   (although this guy's version isn't 
 
 I'm tying all of these metrics back to the world tick in the XDB, so we can report on them in the EU reporting schema per token.  So when we detect and steer a safety concept its got this kind of rigor wrapped around it by defaultgeneral enough)