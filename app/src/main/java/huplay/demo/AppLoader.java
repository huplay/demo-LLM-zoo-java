package huplay.demo;

import huplay.demo.config.TransformerType;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.util.Util;
import huplay.demo.config.Arguments;
import huplay.demo.config.Config;

import java.io.*;
import java.util.*;

import static huplay.demo.AppMain.displayConfig;

public class AppLoader
{
    private static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();

    public static final String ARG_CALC = "-calc";
    public static final String ARG_NO_EXIT = "-noExit";
    public static final String ARG_MAX = "-max";
    public static final String ARG_TOP_K = "-topK";
    public static final String ARG_MEM = "-mem";

    public static void main(String... args) throws Exception
    {
        logo();

        boolean isContinue = true;
        while (isContinue)
        {
            // Read arguments (possibly asking the user to select the model)
            Arguments arguments = readArguments(args);
            isContinue = arguments.isNoExit();

            // Read config of the selected model
            Config config = new Config(arguments);

            if (config.isCalculationOnly())
            {
                // Calculation only. Display config, parameter size
                config.setCalculationOnly(true);
                TransformerType transformerType = TransformerType.valueOf(config.getTransformerType());
                BaseTransformer transformer = transformerType.getTransformer(config);
                displayConfig(config, transformer.getParameterSize());
            }
            else
            {
                // Determine memory requirement
                int memorySize = determineMemoryRequirement(config);

                // Download the parameters if missing
                if (download(config))
                {
                    try
                    {
                        // Open the main app to launch the model
                        String command = "java -Xmx" +
                                memorySize + "m -Xms" + memorySize + "m" +
                                " -cp " + System.getProperty("user.dir") + "/app/target/demo-llm-zoo.jar huplay.demo.AppMain" +
                                " \"" + arguments.getName() + "\"" +
                                " -max=" + config.getLengthLimit() +
                                " -topK=" + config.getTopK();

                        OUT.println("Command:\n" + command + "\n");
                        Runtime.getRuntime().exec("cmd /k start cmd /c " + command);
                    }
                    catch (IOException e)
                    {
                        OUT.println("Error: " + e.getMessage());
                    }
                }
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
        String modelRoot = System.getenv().getOrDefault("DEMO_LLM_ZOO_MODEL_ROOT", configRoot);

        // Default values
        String name = null;
        int maxLength = 25;
        int topK = 40;
        int memorySize = 0;
        boolean isCalculationOnly = false;
        boolean isNoExit = false;

        if (args != null)
        {
            // Iterate over the passed parameters and override the default values
            for (String arg : args)
            {
                if (arg.charAt(0) == '-')
                {
                    if (equals(arg, ARG_CALC)) isCalculationOnly = true;
                    else if (equals(arg, ARG_NO_EXIT)) isNoExit = true;
                    else
                    {
                        String[] parts = arg.split("=");
                        if (parts.length == 2)
                        {
                            String key = parts[0];
                            String value = parts[1];

                            if (equals(key, ARG_MAX)) maxLength = readInt(value, maxLength);
                            else if (equals(key, ARG_TOP_K)) topK = readInt(value, topK);
                            else if (equals(key, ARG_MEM)) memorySize = readInt(value, 0);
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
                    name = removeDoubleQuotes(arg);
                }
            }
        }

        if (name == null)
        {
            String path = ask(configRoot, configRoot);
            name = path.substring(configRoot.length() + 1);
        }

        return new Arguments(name, configRoot, modelRoot, maxLength, topK, isCalculationOnly, isNoExit, memorySize);
    }

    private static boolean equals(String a, String b)
    {
        return a.toLowerCase(Locale.ROOT).equals(b.toLowerCase(Locale.ROOT));
    }

    private static String ask(String path, String configRoot) throws IOException
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
            Map<Integer, String> directories = new LinkedHashMap<>();
            int i = 1;
            int j = -1;
            int length = 1;
            for (File file : files)
            {
                String name = file.getName();
                if (file.isDirectory())
                {
                    if (isEnabled(name))
                    {
                        directories.put(i, name);
                        length = String.valueOf(i).length();
                        i++;
                    }
                    else
                    {
                        directories.put(j, name);
                        j--;
                    }
                }
            }

            if (directories.isEmpty())
            {
                // Go back a level if there's no model here and no subfolders
                OUT.println("There is no model in the selected folder.");
                return ask(getParentFolder(path), configRoot);
            }
            else
            {
                if (!path.equals(configRoot))
                {
                    OUT.println(alignRight("0", length) + ": ..");
                }

                // Display the list of directories
                for (Map.Entry<Integer, String> entry : directories.entrySet())
                {
                    Integer key = entry.getKey();
                    String id = alignRight((key > 0) ? entry.getKey().toString() : "-", length);

                    String displayName = getDisplayName(entry.getValue());

                    OUT.println(id + ": " + displayName);
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
                        if (text.equals("x") || text.equals("X"))
                        {
                            choice = -1;
                            break;
                        }

                        choice = Integer.parseInt(text);
                        if ( (choice > 0 && choice <= directories.size()) || (!path.equals(configRoot) && choice == 0))
                        {
                            break;
                        }

                        OUT.println("Incorrect choice. (Press X to exit any time.)");
                    }
                    catch (Exception e)
                    {
                        OUT.println("Incorrect choice. (Press X to exit any time.)");
                    }
                }

                String newPath = "";
                if (choice == -1)
                {
                    OUT.println("Bye!");
                    System.exit(0);
                }
                else if (choice == 0)
                {
                    newPath = getParentFolder(path);
                }
                else
                {
                    newPath = path + "/" + directories.get(choice);
                }

                OUT.println();
                return ask(newPath, configRoot);
            }
        }

        OUT.println("There are no configured models.");
        OUT.println("Bye!");
        System.exit(0);
        return null;
    }

    private static boolean isEnabled(String name)
    {
        if (name.startsWith("("))
        {
            // Remove the bracketed order from the name
            int closing = name.indexOf(")");

            return closing <= 2 || name.charAt(closing - 2) != '-' || name.charAt(closing - 1) != '-';
        }

        return true;
    }

    private static String getDisplayName(String name)
    {
        if (name.startsWith("("))
        {
            // Remove the bracketed order from the name
            int closing = name.indexOf(")");
            if (closing > 0) name = name.substring(closing + 1);
        }

        return name;
    }

    private static String alignRight(String text, int length)
    {
        if (text.length() >= length) return text;

        char[] pad = new char[length - text.length()];
        Arrays.fill(pad, ' ');
        return new String(pad) + text;
    }

    private static String getParentFolder(String path)
    {
        int lastIndex = path.lastIndexOf("/");
        return path.substring(0, lastIndex);
    }

    private static String removeDoubleQuotes(String text)
    {
        if (text == null) return null;
        if (text.charAt(0) == '"') text = text.substring(1);
        if (text.charAt(text.length() - 1) == '"') text = text.substring(0, text.length() - 1);
        return text;
    }

    private static int determineMemoryRequirement(Config config)
    {
        // First, use the requested memory size (if exists)
        int memorySize = config.getMemorySize();
        if (memorySize <= 0)
        {
            // Second, use the configured total memory size (if exists)
            memorySize = config.getMemorySizeTotal();
            if (memorySize <= 0)
            {
                // Third, use the configured additional memory size (if exists), or the default 2000M
                int additionalMemorySize = config.getMemorySizeAdditional();
                if (additionalMemorySize <= 0)
                {
                    additionalMemorySize = 2000;
                }

                // Calculate the parameter size
                config.setCalculationOnly(true);
                TransformerType transformerType = TransformerType.valueOf(config.getTransformerType());
                BaseTransformer transformer = transformerType.getTransformer(config);
                int parameterMemorySize = Math.round((float) transformer.getParameterSize() / 1000 / 1000 * 4);

                memorySize = additionalMemorySize + parameterMemorySize;
            }
        }

        return memorySize;
    }

    private static boolean download(Config config) throws IOException, InterruptedException
    {
        boolean isOk = true;

        for (String fileName : config.getParameterFiles())
        {
            String path = config.getModelPath() + fileName;
            File file = new File(path);

            if (!file.exists())
            {
                isOk = false;
                OUT.println("Parameter file is missing. (" + fileName + ")");
                if (config.getParameterRepo() != null)
                {
                    OUT.print("Do you want me to download? Repo: " + config.getParameterRepo() + "\nYes or no? ");
                    BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                    String text = reader.readLine();
                    OUT.println();

                    if (text.equals("y") || text.equals("Y"))
                    {
                        Downloader downloader = new Downloader(config, fileName, path);
                        Thread thread = new Thread(downloader);
                        thread.start();

                        int pos = 1;
                        while (downloader.isInProgress())
                        {
                            String progress = "";
                            for (int i = 1; i <= 25; i++)
                            {
                                if (pos == i) progress += "=";
                                else progress += " ";
                            }

                            pos++;
                            if (pos > 25) pos = 1;

                            System.out.print("Downloading |" + progress + "|\r");
                            Thread.sleep(100);
                        }

                        isOk = downloader.isOk();
                    }
                }
            }
        }

        return isOk;
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
