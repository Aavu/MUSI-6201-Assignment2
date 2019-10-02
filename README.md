# MUSI_6201-Assignment2

#### C2
![Plots](plots.png)


RMS mean v/s RMS std features provide the best seperation between the speech and music. We can clearly draw a boundary that mostly seperates the classes. This distinction can be attributed to the reason that speech has more pauses than music, which increase the RMS std value for speech.

The sf_std v/s sc_std and scr_std v/s zcr_std gives some seperation. But it would be hard to classify just with them as we can see, there are a lot of overlapping datapoints and it is hard to clearly draw a seperating boundary.

In SCR_std vs ZCR_std, the speech has a higher zcr_std values probably because the F0 values for speech are generally higher than the F0 for musical instruments.

We dont think scr_mean vs sc_mean and sf_mean vs zcr_mean are good features pairs when used alone as they dont distinguish the data points so well.

But if we had to infer from the plot scr_mean vs sc_mean, we could say that speech seems to have higher SCR values for corresponding SC values, which probably means that speech has a peakier spectrum than music.
