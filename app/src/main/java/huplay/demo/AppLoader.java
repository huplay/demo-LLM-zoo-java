package huplay.demo;

import huplay.demo.config.ModelConfig;
import huplay.demo.config.ParameterReader;
import huplay.demo.transformer.TransformerType;
import huplay.demo.transformer.BaseTransformer;
import huplay.demo.util.Util;
import huplay.demo.config.Arguments;
import huplay.demo.config.Config;

import java.io.*;
import java.util.*;

import static huplay.demo.AppMain.displayConfig;
import static huplay.demo.AppMain.getPrintStream;
import static huplay.demo.Logo.showLogo;
import static java.lang.Math.round;

public class AppLoader
{
    public static final PrintStream OUT = getPrintStream();
    public static final Util UTIL = new Util();

    public static void main(String... args)
    {
        try
        {
            showLogo(OUT,"Demo LLM zoo", "1,2,3,4,,5,6,7,,8,9,10");
            OUT.println("Util: " + UTIL.getUtilName() + "\n");
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
        ModelConfig modelConfig = ModelConfig.read(arguments);

        // Check necessary files
        List<String> missingFiles = checkFiles(modelConfig, arguments.getModelPath());

        // Download the missing files
        download(missingFiles, modelConfig, arguments.getModelPath());

        ParameterReader reader = new ParameterReader(arguments.getModelPath());

        // Read the config (first look into the model folder, second to the config folder (maybe it's different)
        Config config = Config.readConfig(arguments, modelConfig, reader);

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

                OUT.println("Command:\n" + command + "\n");
                Runtime.getRuntime().exec("cmd /k start cmd /c " + command);
            }
            catch (IOException e)
            {
                OUT.println("Error launching the main app: " + e.getMessage());
            }
        }
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
                OUT.println("There is no model in the selected folder.");
                return selectModel(getParentFolder(path), configRoot);
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
                return selectModel(newPath, configRoot);
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
                int parameterMemorySize = round((float) transformer.getParameterSize() / 1000 / 1000 * 4);

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
                OUT.println("Parameter files are missing. " + missingFiles);
                OUT.print("Do you want to download these using the configured url: " + modelConfig.getRepo() + "\nYes or no? ");
                BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
                String text = reader.readLine();
                OUT.println();

                if (text.equals("y") || text.equals("Y"))
                {
                    for (String missingFile : missingFiles)
                    {
                        Downloader downloader = new Downloader(modelConfig, missingFile, modelPath);

                        OUT.println("File: " + missingFile + " (size: " + formatSize(downloader.getSize()) + ")");

                        Thread thread = new Thread(downloader);
                        thread.start();

                        while (downloader.isInProgress())
                        {
                            if (downloader.getPieces() > 0)
                            {
                                long pieces = downloader.getPieces();
                                long position = downloader.getPosition();

                                showProgressBar(pieces, position, 50);

                                Thread.sleep(200);
                            }
                        }

                        // Display a completed progress bar
                        showProgressBar(downloader.getPieces(), downloader.getPieces(), 50);
                    }
                }
                else
                {
                    throw new IdentifiedException("There are missing files: " + missingFiles);
                }
            }
        }
    }

    private String formatSize(long size)
    {
        long x = 1024;

        if (size < x) return size + " Bytes";
        else if (size < x*x) return round((float)size/x) + " kB";
        else if (size < x*x*x) return round((float)size/x/x) + " MB";
        else if (size < x*x*x*x) return round((float)size/x/x/x) + " GB";
        else return round((float)size/x/x/x/x) + " TB";
    }

    // ANSI escape codes
    private static final String BLUE = "\033[0;34m"; // Changes the colour to blue
    private static final String BLUE_BACKGROUND = "\033[44m"; // Changes the background colour to blue
    private static final String WHITE_BACKGROUND = "\033[47m"; // Changes the background colour to white
    private static final String RESET = "\033[0m"; // Changes the colour and background colour to default

    // There's no full block unicode character (0x2588 only almost full),
    // so I created it, changing the background colour of a space character
    private static final String FULL = BLUE_BACKGROUND + " " + WHITE_BACKGROUND;

    // Blocks with growing size (to make the progress more fine-grained)
    private static final char[] BLOCKS = new char[] {' ', 0x258F, 0x258E, 0x258D, 0x258B, 0x258A, 0x2589};

    public void showProgressBar(long total, long actual, int length)
    {
        if (total == actual)
        {
            // Display a completed progress bar
            StringBuilder progressBar = new StringBuilder();
            for (int i = 0; i < length; i++)
            {
                progressBar.append(FULL);
            }

            OUT.print("Download: " + BLUE + progressBar + RESET + " DONE   \n\n"); // Jump to next line
        }
        else
        {
            // Calculate the actual position within the progress bar and the percentage
            // Make it 7 times bigger to be able to show the progress within a character
            long position = Math.floorDiv(length * actual * 7, total);
            String percentage = String.format("%.2f", (float) 100 * actual / total);

            // Calculate the number of full blocks and the size of the last (progressing) character
            long intPos = Math.floorDiv(position, 7);
            long remainderPos = position % 7;

            StringBuilder progressBar = new StringBuilder();
            for (int i = 0; i < length; i++)
            {
                if (intPos > i)
                {
                    progressBar.append(FULL); // Display the full characters
                }
                else if (intPos == i)
                {
                    progressBar.append(BLOCKS[(int) remainderPos]); // Display the actually progressing character
                }
                else
                {
                    progressBar.append(" "); // Display the empty characters
                }
            }

            // Use only \r to remain in the same line and redraw it next time
            OUT.print("Download: " + BLUE + WHITE_BACKGROUND + progressBar + RESET + " " + percentage + "%\r");
        }
    }
}
