package huplay.demo;

import huplay.demo.config.*;
import huplay.demo.tokenizer.Token;
import huplay.demo.tokenizer.Tokenizer;
import huplay.demo.tokenizer.TokenizerType;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.TransformerType;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.AppLoader.checkFiles;

public class AppMain
{
    public static final PrintStream OUT = getPrintStream();

    public static void main(String... args)
    {
        try
        {
            Logo.showLogo(OUT, "Demo LLM zoo", "1,2,3,4,,5,6,7,,8,9,10");
            OUT.println("Util: " + UTIL.getUtilName() + "\n");
            new AppMain().start(args);
        }
        catch (IdentifiedException e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
        catch (Throwable e)
        {
            StackTraceElement[] stackTraceElements = e.getStackTrace();
            if (stackTraceElements != null)
            {
                for (StackTraceElement element : stackTraceElements)
                {
                    OUT.println(element.toString());
                }
            }
            OUT.println("ERROR: " + e.getMessage() + " " + Arrays.toString(e.getStackTrace()));
        }
    }

    private void start(String... args) throws Exception
    {
        // Read arguments
        Arguments arguments = Arguments.readArguments(args);

        // Read the modelConfig of the selected model
        ModelConfig modelConfig = ModelConfig.read(arguments);

        // Check necessary files
        List<String> missingFiles = checkFiles(modelConfig, arguments.getModelPath());
        if (missingFiles.size() > 0)
        {
            throw new IdentifiedException("There are missing files: " + missingFiles);
        }

        // Create the parameter reader
        ParameterReader reader = new ParameterReader(arguments.getModelPath());

        // Read the config (first look into the model folder, second to the config folder (maybe it's different)
        Config config = Config.readConfig(arguments, modelConfig, reader);

        displayConfig(config, 0);

        OUT.print("\nLoading parameters... ");
        BaseTransformer transformer = TransformerType.getTransformer(config);

        OUT.println("Done.");
        OUT.println("Parameter size:  " + Math.round((float) transformer.parameterSize / 1000_000) + "M");

        if (!config.isCalculationOnly())
        {
            Tokenizer tokenizer = TokenizerType.getTokenizer(config);
            Generate processor = new Generate(config, tokenizer, transformer);

            int pos = 0;
            int lastToken = config.getEndOfTextToken();

            while (true)
            {
                // Read the input text
                String inputText = input();

                List<Integer> inputTokens = new ArrayList<>();

                if (inputText == null)
                {
                    break;
                }
                else if (inputText.equals("+"))
                {
                    // If the input is "+", continue the generation as usual (adding the last output to the input)
                    inputTokens.add(lastToken);
                }
                else
                {
                    if (inputText.startsWith("+"))
                    {
                        // Input starts with "+" is a request to continue the same session
                        // Remove the "+", and don't clear the position and stored values
                        inputText = inputText.substring(1);
                    }
                    else
                    {
                        // Clear the transformer's stored values
                        pos = 0;
                        processor.clear();
                    }

                    // Convert the input text into list of tokens
                    inputTokens.addAll(tokenizer.encode(inputText));

                    // Display the coloured version of the input to show the tokens
                    List<Token> split = tokenizer.split(inputText);
                    OUT.print("            ");
                    for (int i = 0; i < split.size(); i++)
                    {
                        String color = i % 2 == 0 ? "\033[32m" : "\033[33m";
                        OUT.print(color + split.get(i).getText().replace("\n", "").replace("\r", ""));
                    }
                    OUT.println("\033[0m");
                }

                // Use the Transformer
                List<Integer> outputTokens = processor.process(inputTokens, pos);

                // Convert the output to text and print it
                String response = tokenizer.decode(outputTokens);
                print(response, outputTokens, tokenizer);

                pos += outputTokens.size();
                lastToken = outputTokens.get(outputTokens.size() - 1);
            }
        }
    }

    public static void displayConfig(Config config, long parameterSize)
    {
        // Print settings
        OUT.println("Model: " + config.getName());
        OUT.println("Path: " + config.getModelPath());
        if (parameterSize > 0)
        {
            OUT.print("Number of parameters: " + Math.round(parameterSize / 1000_000d) + "M ");
        }
        OUT.println("Hidden size: " + config.getHiddenSize() +
                ", decoders: " + config.getDecoderCount() +
                ", heads: " + config.getHeadCount() +
                ", head size: " + config.getHeadSize());

        OUT.println("Maximum length of generated text: " + config.getLengthLimit());
        OUT.println("Output is selected from the best " + config.getTopK() + " tokens (topK)");
        OUT.println("Max memory: " + config.getMemorySize());
    }

    private static String input() throws IOException
    {
        OUT.print("\n\nInput text: ");
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        return reader.readLine();
    }

    private static void print(String response, List<Integer> outputTokens, Tokenizer tokenizer)
    {
        // The response was printed token by token, but for multi-token characters only "�" will be displayed

        // Here we recreate the token by token decoded response (which wasn't returned)
        StringBuilder tokenByTokenResponse = new StringBuilder();
        for (int token: outputTokens)
        {
            tokenByTokenResponse.append(tokenizer.decode(Collections.singletonList(token)));
        }

        // If the token by token decoded result is different to the final decoded result, print the corrected version
        if ( ! tokenByTokenResponse.toString().equals(response))
        {
            OUT.print("\nCorrected unicode response:\n" + response);
        }
    }

    public static PrintStream getPrintStream()
    {
        try
        {
            return new PrintStream(System.out, true, StandardCharsets.UTF_8);
        }
        catch (Exception e)
        {
            System.out.println("\nError during setting the console to UTF-8:\n" + e.getMessage());
            return System.out;
        }
    }
}
