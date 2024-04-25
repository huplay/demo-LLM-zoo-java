# Demo (almost) all LLM architectures in Java

This is a demo application which implements different LLM (Large Language Model) architectures like GPT-1, GPT-2 and Llama in Java, for learning purposes.

The goal is to demonstrate the different decoder-only transformer architectures (without training), not to create an optimized application.

TensorFlow, Pytorch or similar tools are NOT used. The core mathematical utility is implemented in three versions, you can select which one to use.

## Trained parameters ##

To use this app you have to find the trained parameters in safetensors format.

There is a `modelConfig` folder where all the ported models have a subfolder with a configuration file.

## Install ##

1. Install Java. For the standard version 1.8 or above. For the Vector API implementation at least Java 18. (Tested only on Java 20).


2. Install Maven. (Java compile/build tool) (3.8.6 used during development).


3. Download and unzip this module: https://github.com/huplay/demo-LLM-zoo-java

   (Or using git: ```git clone https://github.com/huplay/demo-LLM-zoo-java.git```)


4. Download and unzip the files with the trained parameters for the version you want to use.

   The files should be placed into the `modelConfig/<model name>` folder, so for example using the GPT-2 XL version to `modelConfig/GPT2/XL`. 

5. Using a command line tool (`cmd`) enter into the main directory:
   
    ```cd demo-LLM-zoo-java```


6. Compile (build) the application. There are 3 possibilities, based on that which utility implementation you want to use.
   Standard: 

   ```mvn clean install -Pstandard```

   Using Nd4j:

   ```mvn clean install -Pnd4j```

   Using Java Vector API:

   ```mvn clean install -Pvector-api```


## Execution ##

Execute the application:
```run <model-name>``` (On Windows)
    
Or on any systems:```java -jar target/demo-LLM-zoo.jar <model-name>```

The models are organized in a folder structure, so somtimes the `model-name` should contain its path. For example:

`run GPT1`

`run GPT2/SMALL`

`run Llama2\tinyLlama15M`


If you want to use the Vector API version (in the case you installed that variant) you have to use the ``runv <model-name>`` command.
This is necessary because the Vector API isn't ready (as of Java 20), added only as an incubator module, so we have to execute the Java Virtual Machine telling we want to use this incubator feature. 
  
Using larger models it is necessary to increase the heap size (memory for Java). The ```run.bat / runv.bat``` handles it automatically, but if the app is called directly you should use the Java -Xmx and Xms flags. 


## Additional command line parameters ##

- `config-root` - Path of the `modelConfig` folder (default: `/modelConfig`)
- `model-root` - Path of the parameters folder (it can be different to the config-root) (default: `/modelConfig`)
- `max` - Maximum number of generated tokens (default: 25)
- `topk` - Number of possibilities to chose from as next token (default: 40)

Example:

`run GPT2/XL max=1024 topk=100`

## Usage ##

The app shows a prompt, where you can provide a text:

```Input text:```

You can leave it empty, or type something, which will be continued by the system. While the input tokens are processed a `.` character is displayed. (One for every token.)
After that the system prints the generated tokens (one by one). If the maximum length is reached, or the response finished by an `END-OF-TEXT` token, a new prompt will be given.

Normally every prompt starts a completely new session (the state is cleared), but if you want to remain in the same context, start your input text by `+`.
If you use only a single `+` character, without more content, the system will continue the text as it would do without the limit of the max length.

To quit press Ctrl + C.

If the response contained special unicode characters, where a single character is constructed using multiple tokens, then the "one by one" printing solution will show "?" characters. But after the text is fully generated the whole text will be printed again to show the correct characters. (Only at cases when the original print wasn't perfect.) 


## Tokenizer ##

All LLMs use a byte pair encoding logic, but there are different versions.

Supported tokenizers: 
   - GPT-1
   - GPT-2 (used by GPT-3 as well), and for BLOOM with different vocabulary
   - SentencePiece (used by Llama1 and Llama2)
