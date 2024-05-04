package huplay.demo;

import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.demo.config.ModelConfig;
import huplay.demo.transformer.TransformerType;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.util.Util;
import huplay.demo.config.Arguments;
import huplay.demo.config.Config;

import java.io.*;
import java.util.*;

import static huplay.demo.AppMain.displayConfig;
import static huplay.demo.AppMain.getPrintStream;

public class AppLoader
{
    public static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();
    private static final ObjectMapper objectMapper = new ObjectMapper();

    public static void main(String... args)
    {
        try
        {
            logo();
            new AppLoader().start(args);
        }
        catch (Exception e)
        {
            OUT.println("ERROR: " + e.getMessage());
        }
    }

    private void start(String... args) throws Exception
    {
        // Read arguments
        Arguments arguments = Arguments.readArguments(args);

        // If the model isn't specified, allow the user to select it
        if (arguments.getConfigPath() == null)
        {
            selectModel(arguments);
        }

        // Read the modelConfig of the selected model
        ModelConfig modelConfig = ModelConfig.read(arguments, objectMapper);

        // Check necessary files
        List<String> missingFiles = checkFiles(modelConfig, arguments.getModelPath());

        // Download the missing files
        download(missingFiles, modelConfig, arguments.getModelPath());

        // Read the config (first look into the model folder, second to the config folder (maybe it's different)
        Config config = Config.read(arguments, modelConfig, objectMapper);

        if (arguments.isCalculationOnly())
        {
            // Calculation only. Display config, parameter size
            config.setCalculationOnly(true);
            BaseTransformer transformer = TransformerType.getTransformer(config);
            displayConfig(config, transformer.getParameterSize());
        }
        else
        {
            // Determine memory requirement
            int memorySize = determineMemoryRequirement(config);

            try
            {
                String userDir = System.getProperty("user.dir").replace('\\', '/');

                // Open the main app to launch the model
                String command = "java" +
                                " -Xmx" + memorySize + "m -Xms" + memorySize + "m" +
                                " -cp " + userDir + "/app/target/demo-llm-zoo.jar" +
                                " huplay.demo.AppMain" +
                                " \"" + arguments.getRelativePath() + "\"" +
                                " -max=" + config.getLengthLimit() +
                                " -topK=" + config.getTopK();

                System.out.println("Command:\n" + command + "\n");
                Runtime.getRuntime().exec("cmd /k start cmd /c " + command);
            }
            catch (IOException e)
            {
                System.out.println("Error launching the main app: " + e.getMessage());
            }
        }
    }

    public static void logo()
    {
        System.out.println(" ____                          _     _     __  __");
        System.out.println("|  _ \\  ___ _ __ ___   ___    | |   | |   |  \\/  |   _______   ___");
        System.out.println("| | | |/ _ \\ '_ ` _ \\ / _ \\   | |   | |   | |\\/| |  |_  / _ \\ / _ \\");
        System.out.println("| |_| |  __/ | | | | | (_) |  | |___| |___| |  | |   / / (_) | (_) |");
        System.out.println("|____/ \\___|_| |_| |_|\\___/   |_____|_____|_|  |_|  /___\\___/ \\___/");
        System.out.println("Util: " + UTIL.getUtilName() + "\n");
    }

    private void selectModel(Arguments arguments) throws Exception
    {
        String configRoot = arguments.getConfigRoot();
        String configPath = selectModel(configRoot, configRoot);
        String relativePath = configPath.substring(configRoot.length() + 1);

        arguments.setRelativePath(relativePath);
    }

    private String selectModel(String path, String configRoot) throws IOException
    {
        File[] fileList = new File(path).listFiles();

        if (fileList != null)
        {
            List<File> files = Arrays.asList(fileList);
            Collections.sort(files);

            // Find model.properties
            for (File file : files)
            {
                if (file.isFile() && file.getName().equals("model.json"))
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
                System.out.println("There is no model in the selected folder.");
                return selectModel(getParentFolder(path), configRoot);
            }
            else
            {
                if (!path.equals(configRoot))
                {
                    System.out.println(alignRight("0", length) + ": ..");
                }

                // Display the list of directories
                for (Map.Entry<Integer, String> entry : directories.entrySet())
                {
                    Integer key = entry.getKey();
                    String id = alignRight((key > 0) ? entry.getKey().toString() : "-", length);

                    String displayName = getDisplayName(entry.getValue());

                    System.out.println(id + ": " + displayName);
                    i++;
                }

                // Ask user to select (repeat at incorrect selection)
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

                int choice;
                while (true)
                {
                    System.out.print("Please select: ");
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

                        System.out.println("Incorrect choice. (Press X to exit any time.)");
                    }
                    catch (Exception e)
                    {
                        System.out.println("Incorrect choice. (Press X to exit any time.)");
                    }
                }

                String newPath = "";
                if (choice == -1)
                {
                    System.out.println("Bye!");
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

                System.out.println();
                return selectModel(newPath, configRoot);
            }
        }

        System.out.println("There are no configured models.");
        System.out.println("Bye!");
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

    private static int determineMemoryRequirement(Config config)
    {
        // First, use the requested memory size (if exists)
        Integer memorySize = config.getRequestedMemorySize();
        if (memorySize == 0)
        {
            // Second, use the configured total memory size (if exists)
            memorySize = config.getMemorySize();
            if (memorySize == null || memorySize == 0)
            {
                // Third, calculate the required memory
                config.setCalculationOnly(true);
                BaseTransformer transformer = TransformerType.getTransformer(config);
                int parameterMemorySize = Math.round((float) transformer.getParameterSize() / 1000 / 1000 * 4);

                memorySize = parameterMemorySize + 2048;
            }
        }

        return memorySize;
    }

    public static List<String> checkFiles(ModelConfig modelConfig, String modelPath)
    {
        List<String> missingFiles = new ArrayList<>();

        for (String fileName : modelConfig.getFiles())
        {
            String path = modelPath + "/" + fileName;

            File file = new File(path);
            if (!file.exists())
            {
                missingFiles.add(fileName);
            }
        }

        return missingFiles;
    }

    private void download(List<String> missingFiles, ModelConfig modelConfig, String modelPath) throws Exception
    {
        if (missingFiles.size() > 0)
        {
            if (modelConfig.getRepo() == null || modelConfig.getRepo().equals(""))
            {
                throw new IdentifiedException("There are missing files: " + missingFiles);
            }
            else
            {
                System.out.println("Parameter files are missing. " + missingFiles);
                System.out.print("Do you want me to download these? Repo: " + modelConfig.getRepo() + "\nYes or no? ");
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                String text = reader.readLine();
                System.out.println();

                if (text.equals("y") || text.equals("Y"))
                {
                    for (String missingFile : missingFiles)
                    {
                        System.out.println("Downloading files to " + modelPath);
                        Downloader downloader = new Downloader(modelConfig, missingFile, modelPath);
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
                    }
                }
                else
                {
                    throw new IdentifiedException("There are missing files: " + missingFiles);
                }
            }
        }
    }
}
