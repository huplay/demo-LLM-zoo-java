package huplay.demo;

import huplay.demo.config.TransformerType;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.util.Util;
import huplay.demo.config.Arguments;
import huplay.demo.config.Config;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static huplay.demo.AppMain.displayConfig;

public class AppLoader
{
    private static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();

    public static final String ARG_CALC = "-calc";
    public static final String ARG_MAX = "-max";
    public static final String ARG_TOP_K = "-topk";
    public static final String ARG_MEM = "-mem";

    public static void main(String... args) throws Exception
    {
        logo();

        Arguments arguments = readArguments(args);
        Config config = new Config(arguments);

        // TODO: Display all
        if (config.isCalculationOnly())
        {
            config.setCalculationOnly(true);
            TransformerType transformerType = TransformerType.valueOf(config.getTransformerType());
            BaseTransformer transformer = transformerType.getTransformer(config);

            displayConfig(config, transformer.getParameterSize());
        }
        else
        {
            int memorySize = config.getMemorySize();
            if (memorySize <= 0)
            {
                memorySize = config.getMemorySizeOverride();
                if (memorySize <= 0)
                {
                    int baseMemorySize = config.getBaseMemorySize();
                    if (baseMemorySize <= 0)
                    {
                        baseMemorySize = 2000;
                    }

                    config.setCalculationOnly(true);
                    TransformerType transformerType = TransformerType.valueOf(config.getTransformerType());
                    BaseTransformer transformer = transformerType.getTransformer(config);

                    memorySize = baseMemorySize + Math.round((float) transformer.getParameterSize() / 1000 / 1000 * 4);
                }
            }

            try
            {
                String command = "java -Xmx" + memorySize + "m -Xms" + memorySize + "m " +
                        "-cp " + System.getProperty("user.dir") + "/app/target/demo-llm-zoo.jar huplay.demo.AppMain " +
                        arguments.getName() + " -max=" + config.getLengthLimit() + " -topk=" + config.getTopK();

                OUT.println("Command: " + command);
                Runtime.getRuntime().exec("cmd /k start cmd /c " + command);
            }
            catch (IOException e)
            {
                OUT.println("Error: " + e.getMessage());
            }
        }
    }

    public static void logo()
    {
        OUT.println(" ____                          _     _     __  __");
        OUT.println("|  _ \\  ___ _ __ ___   ___    | |   | |   |  \\/  |   _______   ___");
        OUT.println("| | | |/ _ \\ '_ ` _ \\ / _ \\   | |   | |   | |\\/| |  |_  / _ \\ / _ \\");
        OUT.println("| |_| |  __/ | | | | | (_) |  | |___| |___| |  | |   / / (_) | (_) |");
        OUT.println("|____/ \\___|_| |_| |_|\\___/   |_____|_____|_|  |_|  /___\\___/ \\___/");
        OUT.println("Util: " + UTIL.getUtilName() + "\n");
    }

    public static Arguments readArguments(String[] args) throws Exception
    {
        File file = new File("modelConfig");
        String configRoot = file.getAbsolutePath();
        String modelRoot = System.getenv().getOrDefault("DEMO_GPT_MODEL_ROOT", configRoot);

        // Default values
        String name = null;
        int maxLength = 25;
        int topK = 40;
        int memorySize = 0;
        boolean isCalculationOnly = false;

        if (args != null)
        {
            // Iterate over the passed parameters and override the default values
            for (String arg : args)
            {
                if (arg.charAt(0) == '-')
                {
                    if (arg.equals(ARG_CALC)) isCalculationOnly = true;
                    else
                    {
                        String[] parts = arg.split("=");
                        if (parts.length == 2)
                        {
                            String key = parts[0];
                            String value = parts[1];

                            if (key.equals(ARG_MAX)) maxLength = readInt(value, maxLength);
                            else if (key.equals(ARG_TOP_K)) topK = readInt(value, topK);
                            else if (key.equals(ARG_MEM)) memorySize = readInt(value, 0);
                        }
                        else
                        {
                            OUT.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                        }
                    }
                }
                else if (name != null)
                {
                    OUT.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                }
                else
                {
                    name = arg;
                }
            }
        }

        if (name == null)
        {
            String path = ask(configRoot);
            name = path.substring(configRoot.length() + 1);
        }

        return new Arguments(name, configRoot, modelRoot, maxLength, topK, isCalculationOnly, memorySize);
    }

    private static String ask(String path) throws IOException
    {
        File[] fileList = new File(path).listFiles();

        if (fileList != null)
        {
            List<File> files = Arrays.asList(fileList);
            Collections.sort(files);

            // Find model.properties
            for (File file : files)
            {
                if (file.isFile() && file.getName().equals("model.properties"))
                {
                    return path;
                }
            }

            // Find directories
            List<String> directories = new ArrayList<>();
            for (File file : fileList)
            {
                if (file.isDirectory())
                {
                    directories.add(file.getName());
                }
            }

            if (directories.isEmpty())
            {
                OUT.println("There are no models in this folder");
                System.exit(0);
            }
            else if (directories.size() == 1)
            {
                // If there's only a single possibility, jump into that
                return ask(path + "/" + directories.get(0));
            }
            else
            {
                // Display the list of directories
                int i = 1;
                for (String directory : directories)
                {
                    String name = directory;
                    if (directory.startsWith("("))
                    {
                        int closing = directory.indexOf(")");
                        name = directory.substring(closing + 1);
                    }

                    OUT.println(i + ": " + name);
                    //directories.add(directory);
                    i++;
                }

                // Ask user to select (repeat at incorrect selection)
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

                int choice;
                while (true)
                {
                    OUT.print("Please select: ");
                    String text = reader.readLine();

                    try
                    {
                        choice = Integer.parseInt(text);
                        if (choice > 0 && choice <= directories.size())
                        {
                            break;
                        }

                        OUT.print("Incorrect choice. ");
                    }
                    catch (Exception e)
                    {
                        OUT.print("Incorrect choice. ");
                    }
                }

                // Use the selected directory
                OUT.println();
                return ask(path + "/" + directories.get(choice - 1));
            }
        }

        OUT.println("There are no models in this folder");
        System.exit(0);

        return null;
    }

    private static int readInt(String value, int defaultValue)
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            OUT.println("\nWARNING: The provided value can't be converted to integer (" + value
                    + "). Default value will be used.\n");
        }
        return defaultValue;
    }

    public static PrintStream getPrintStream()
    {
        try
        {
            return new PrintStream(System.out, true, "utf-8");
        }
        catch (Exception e)
        {
            System.out.println("\nError during setting the console to UTF-8:\n" + e.getMessage());
            return System.out;
        }
    }
}
