![GitHub issues](https://img.shields.io/github/issues-raw/nalmadi/EMIP-Toolkit?style=for-the-badge)
![GitHub](https://img.shields.io/github/license/nalmadi/EMIP-Toolkit?style=for-the-badge)
![GitHub last commit](https://img.shields.io/github/last-commit/nalmadi/EMIP-Toolkit?style=for-the-badge)
[![Watchers](https://img.shields.io/github/watchers/nalmadi/EMIP-Toolkit?color=magenta)](https://github.com/nalmadi/EMIP-Toolkit)
[![Forks](https://img.shields.io/github/forks/nalmadi/EMIP-Toolkit?color=magenta)](https://github.com/nalmadi/EMIP-Toolkit)
[![Stars](https://img.shields.io/github/stars/nalmadi/EMIP-Toolkit?color=magenta)](https://github.com/nalmadi/EMIP-Toolkit)


# 👀 Eye Movement In Programming Toolkit (EMTK)
EMIP-Toolkit (EMTK): A Python Library for Processing Eye Movement in Programming Data


The use of eye tracking in the study of program comprehension in software engineering allows researchers to gain a better understanding of the strategies and processes applied by programmers. Despite the large number of eye tracking studies in software engineering, very few datasets are publicly available. 

# 💾 Datasets:
This tool evolved to include the following datasets:
1. **EMIP2020**: Bednarik, Roman, et al. "EMIP: The eye movements in programming dataset." Science of Computer Programming 198 (2020): 102520.
2. **AlMadi2018**: Al Madi, Naser, and Javed Khan. "Constructing semantic networks of comprehension from eye-movement during reading." 2018 IEEE 12th International Conference on Semantic Computing (ICSC). IEEE, 2018.
3. **McChesney2021**: McChesney, Ian, and Raymond Bond. "Eye Tracking Analysis of Code Layout, Crowding and Dyslexia-An Open Data Set." ACM Symposium on Eye Tracking Research and Applications. 2021.
4. **AlMadi2021**: Al Madi, Naser, et al. "EMIP Toolkit: A Python Library for Customized Post-processing of the Eye Movements in Programming Dataset." ACM Symposium on Eye Tracking Research and Applications. 2021.

We would be happy to include more eye movement datasets if you have any suggestions.

# 🎥 Presentation:
[![Watch the video](https://imgur.com/IcowLr3.png)](https://www.youtube.com/watch?v=wFdGyM6qUlE)
[Read our paper](https://www.researchgate.net/publication/350485560_EMIP_Toolkit_A_Python_Library_for_Customized_Post-processing_of_the_Eye_Movements_in_Programming_Dataset).


# ⚙️ Features:
The toolkit is designed to make using and processing eye movement in programming datasets easier and more accessible by providing the following functions:
 
 - Parsing raw data files from existing datasets into pandas dataframes.
    
 - Customizable fixation detection algorithms.
   
 - Raw data and filtered data visualizations for each trial.
    
 - Hit testing between fixations and AOIs to determine the fixations over each AOI.
        
 - Customizable offset-based fixation correction implementation for each trial.
    
 - Customizable Areas Of Interest (AOIs) mapping implementation at the line level or token level in source code for each trial.
    
 - Visualizing AOIs before and after fixations overlay on the code stimulus.
    
 - Mapping source code tokens to generated AOIs and eye movement data.
    
 - Adding source code lexical category tags to eye movement data using [srcML](https://www.srcml.org/). srcML is a static analysis tool and data format that provides very accurate syntactic categories (method signatures, parameters, function names, method calls, declarations and so on) for source code. We use it to enhance the eye movements dataset to enable better querying capabilities. 


# ✍️ Examples and tutorial:
The Jupyter Notebook files contain examples and a tutorial on using the EMTK with each dataset.


# 📝 Please Cite This Paper: 
[Naser Al Madi, Drew T. Guarnera, Bonita Sharif, and Jonathan I. Maletic.2021. EMIP Toolkit: A Python Library for Customized Post-processing of the Eye Movements in Programming Dataset. In ETRA ’21: 2021 Symposium on Eye Tracking Research and Applications (ETRA ’21 Short Papers), May25–27, 2021, Virtual Event, Germany. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3448018.3457425](https://www.researchgate.net/publication/350485560_EMIP_Toolkit_A_Python_Library_for_Customized_Post-processing_of_the_Eye_Movements_in_Programming_Dataset)

