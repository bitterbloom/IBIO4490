# Introduction to Linux

## Preparation

1. Boot from a usb stick (or live cd), we suggest to use  [Ubuntu gnome](http://ubuntugnome.org/) distribution, or another ubuntu derivative.

2. (Optional) Configure keyboard layout and software repository
   Go to the the *Activities* menu (top left corner, or *start* key):
      -  Go to settings, then keyboard. Set the layout for latin america
      -  Go to software and updates, and select the server for Colombia
3. (Optional) Instead of booting from a live Cd. Create a partition in your pc's hard drive and install the linux distribution of your choice, the installed Os should perform better than the live cd.

## Introduction to Linux

1. Linux Distributions

   Linux is free software, it allows to do all sort of things with it. The main component in linux is the kernel, which is the part of the operating system that interfaces with the hardware. Applications run on top of it. 
   Distributions pack together the kernel with several applications in order to provide a complete operating system. There are hundreds of linux distributions available. In
   this lab we will be using Ubuntu as it is one of the largest, better supported, and user friendly distributions.


2. The graphical interface

   Most linux distributions include a graphical interface. There are several of these available for any taste.
   (http://www.howtogeek.com/163154/linux-users-have-a-choice-8-linux-desktop-environments/).
   Most activities can be accomplished from the interface, but the terminal is where the real power lies.

### Playing around with the file system and the terminal
The file system through the terminal
   Like any other component of the Os, the file system can be accessed from the command line. Here are some basic commands to navigate through the file system

   -  ``ls``: List contents of current directory
   - ``pwd``: Get the path  of current directory
   - ``cd``: Change Directory
   - ``cat``: Print contents of a file (also useful to concatenate files)
   - ``mv``: Move a file
   - ``cp``: Copy a file
   - ``rm``: Remove a file
   - ``touch``: Create a file, or update its timestamp
   - ``echo``: Print something to standard output
   - ``nano``: Handy command line file editor
   - ``find``: Find files and perform actions on it
   - ``which``: Find the location of a binary
   - ``wget``: Download a resource (identified by its url) from internet 

Some special directories are:
   - ``.`` (dot) : The current directory
   -  ``..`` (two dots) : The parent of the current directory
   -  ``/`` (slash): The root of the file system
   -  ``~`` (tilde) :  Home directory
      
Using these commands, take some time to explore the ubuntu filesystem, get to know the location of your user directory, and its default contents. 
   
To get more information about a command call it with the ``--help`` flag, or call ``man <command>`` for a more detailed description of it, for example ``man find`` or just search in google.


## Input/Output Redirections
Programs can work together in the linux environment, we just have to properly 'link' their outputs and their expected inputs. Here are some simple examples:

1. Find the ```passwd```file, and redirect its contents error log to the 'Black Hole'
   >  ``find / -name passwd  2> /dev/null``

   The `` 2>`` operator redirects the error output to ``/dev/null``. This is a special file that acts as a sink, anything sent to it will disappear. Other useful I/O redirection operations are
      -  `` > `` : Redirect standard output to a file
      -  `` | `` : Redirect standard output to standard input of another program
      -  `` 2> ``: Redirect error output to a file
      -  `` < `` : Send contents of a file to standard input
      -  `` 2>&1``: Send error output to the same place as standard output

2. To modify the content display of a file we can use the following command. It sends the content of the file to the ``tr`` command, which can be configured to format columns to tabs.

   ```bash
   cat milonga.txt | tr '\n' ' '
   ```
   
## SSH - Server Connection

1. The ssh command lets us connect to a remote machine identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER (**vision** in our case). The second command allows us to copy files between systems (you will get the actual login information in class).

   ```bash
   
   #connect
   ssh USER@SERVER
   ```

2. The scp command allows us to copy files form a remote server identified by SERVER (either a name that can be resolved by the DNS, or an ip address), as the user USER. Following the SERVER information, we add ':' and write the full path of the file we want to copy, finally we add the local path where the file will be copied (remember '.' is the current directory). If we want to copy a directory we add the -r option. for example:

   ```bash
   #copy 
   scp USER@SERVER:~/data/sipi_images .
   
   scp -r USER@SERVER:/data/sipi_images .
   ```
   
   Notice how the first command will fail without the -r option

See [here](ssh.md) for different types of SSH connection with respect to your OS.

## File Ownership and permissions   

   Use ``ls -l`` to see a detailed list of files, this includes permissions and ownership
   Permissions are displayed as 9 letters, for example the following line means that the directory (we know it is a directory because of the first *d*) *images*
   belongs to user *vision* and group *vision*. Its owner can read (r), write (w) and access it (x), users in the group can only read and access the directory, while other users can't do anything. For files the x means execute. 
   ```bash
   drwxr-x--- 2 vision vision 4096 ene 25 18:45 images
   ```
   
   -  ``chmod`` change access permissions of a file (you must have write access)
   -  ``chown`` change the owner of a file
   
## Sample Exercise: Image database

1. Create a folder with your Uniandes username. (If you don't have Linux in your personal computer)

2. Copy *sipi_images* folder to your personal folder. (If you don't have Linux in your personal computer)

3.  Decompress the images (use ``tar``, check the man) inside *sipi_images* folder. 

4.  Use  ``imagemagick`` to find all *grayscale* images. We first need to install the *imagemagick* package by typing

    ```bash
    sudo apt-get install imagemagick
    ```
    
    Sudo is a special command that lets us perform the next command as the system administrator
    (super user). In general it is not recommended to work as a super user, it should only be used 
    when it is necessary. This provides additional protection for the system.
    
    ```bash
    find . -name "*.tiff" -exec identify {} \; | grep -i gray | wc -l
    ```
    
3.  Create a script to copy all *color* images to a different folder
    Lines that start with # are comments
       
      ```bash
      #!/bin/bash
      
      # go to Home directory
      cd ~ # or just cd

      # remove the folder created by a previous run from the script
      rm -rf color_images

      # create output directory
      mkdir color_images

      # find all files whose name end in .tif
      images=$(find sipi_images -name *.tiff)
      
      #iterate over them
      for im in ${images[*]}
      do
         # check if the output from identify contains the word "gray"
         identify $im | grep -q -i gray
         
         # $? gives the exit code of the last command, in this case grep, it will be zero if a match was found
         if [ $? -eq 0 ]
         then
            echo $im is gray
         else
            echo $im is color
            cp $im color_images
         fi
      done
      
      ```
      -  save it for example as ``find_color_images.sh``
      -  make executable ``chmod u+x`` (This means add Execute permission for the user)
      -  run ``./find_duplicates.sh`` (The dot is necessary to run a program in the current directory)
      

## Your turn

1. What is the ``grep``command?

The 'grep' command is a command that searches for a certain pattern inside a certain file. If it finds it, it prints it in the terminal. Additionally, you can use wildcards such as * to search for the pattern in multiple files. Such sintax could be grep "Hello" *.txt which finds the pattern "Hello" in any .txt file.

2. What is the meaning of ``#!/bin/python`` at the start of scripts?

The "#!/bin/python" or "#!/bin/bash" are called snippets that are found at the start of scripts. When the user calls an executable file without specifying which program to run it on, the system reads the first line to find if there is a snippet that specifies it for the user. In the first case, it is telling the system to run such file with python and in the second case with bash.

3. Download using ``wget`` the [*bsds500*](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500) image segmentation database, and decompress it using ``tar`` (keep it in you hard drive, we will come back over this data in a few weeks).

As you can see in the image, I downloaded the database using wget.

![wget](https://user-images.githubusercontent.com/45858167/52185608-31579080-27ef-11e9-845d-88828b61fbd0.png)
 
4. What is the disk size of the uncompressed dataset, How many images are in the directory 'BSR/BSDS500/data/images'?

As you can see in the image, the size of the uncompressed dataset is around 73 Mb. I was able to do this by using the command du -sh that I found in this [*link*](https://unix.stackexchange.com/questions/185764/how-do-i-get-the-size-of-a-directory-on-the-command-line)

![diskspace](https://user-images.githubusercontent.com/45858167/52185623-564c0380-27ef-11e9-9e36-cb86134c6f47.png)

One easy way to count how many images are in the images folder is to get inside each folder and use the command 
```bash
find -name "*.jpg" | wc -l 
```
to get the number of files inside the folder. If we do this for the 3 folders in images we find that there are 500 images. 200 in train, 200 in test and 100 in val.

 
5. What are all the different resolutions? What is their format? Tip: use ``awk``, ``sort``, ``uniq`` 

There are 2 kinds of resolutions, 481x321 and 321x481 (landscape and portrait respectively). Their format is .jpg.

To find the format I first used identify in one random image in test. When I saw the jpg format I used 
```bash
find . -name "*.*" -exec identify {} \; | grep -i jpg | wc -l
```
and got a count of 200 which is the same as the number I obtained in point 4 and therefore all the images are .jpg. I did the same thing with train and val and obtained the same result. For the resolutions I used the following code:

```bash
#!/bin/bash

images=$(find -name "*.jpg")
touch reso.txt

for im in ${images[*]}
do
(identify $im | cut -d ' ' -f 3) >> reso.txt
done
```

After I had the reso.txt file I used the sort command and found that there were only two resolutions as described before. I did the same on the three folders and found the same results. 

6. How many of them are in *landscape* orientation (opposed to *portrait*)? Tip: use ``awk`` and ``cut``

I used the following code to find the amount of landscape oriented pictures vs portrait oriented ones:

```bash
#!/bin/bash

images=$(find -name "*.jpg")

declare -i lands=0
declare -i port=0
for im in ${images[*]}
do
identify $im | grep -i -q 481x321
if [ $? -eq 0 ]
then
lands=$[lands+1] 
else
port=$[port+1] 
fi
done

echo lands = $lands
echo port = $port
```

When I did this on each of the folders (test, train, and val) the distribution goes as it follows:  
test -> 134 landscape, 66 portrait  
train-> 137 landscape, 63 portrait  
val  ->  77 landscape, 23 portrait  

For the arithmetic I used I guided myself with the comment of Karoly Horvath in this [*link*](https://stackoverflow.com/questions/6348902/how-can-i-add-numbers-in-a-bash-script)

7. Crop all images to make them square (256x256) and save them in a different folder. Tip: do not forget about  [imagemagick](http://www.imagemagick.org/script/index.php).

First I created a folder inside each of the folders of test, train and val called X_256. After I did this, I copied every file in the folder into the recently created one. This code can be found below:

```bash
#!/bin/bash

images=$(find -name "*.jpg")
rm train_256
mkdir train_256

for im in ${images[*]}
do
cp $im ./train_256
done
```

![new_directory](https://user-images.githubusercontent.com/45858167/52231511-2997fa80-2888-11e9-97cc-aff81ab889e6.png)

When I had all the files in the new folder, I had to resize them to 256x256. To do this I implemented the following code:

```bash
#!/bin/bash

images=$(find -name "*.jpg")
for im in ${images[*]}
do
convert $im -resize 256x256\! $im
done
```

![new_size](https://user-images.githubusercontent.com/45858167/52231515-2c92eb00-2888-11e9-8069-b5caca563b3a.png)

As you can see, all the images are of the same size just by looking at them. For detailed information, I used the following code to check and the image to prove it.

```bash
#!/bin/bash

identify -format "%G" 2092.jpg
```

![prove](https://user-images.githubusercontent.com/45858167/52231519-2f8ddb80-2888-11e9-9523-7362460a3e68.png)

# Report

For every question write a detailed description of all the commands/scripts you used to complete them. DO NOT use a graphical interface to complete any of the tasks. Use screenshots to support your findings if you want to. 

Feel free to search for help on the internet, but ALWAYS report any external source you used.

Notice some of the questions actually require you to connect to the course server, the login instructions and credentials will be provided on the first session. 

## Deadline

We will be delivering every lab through the [github](https://github.com) tool (Silly link isn't it?). According to our schedule we will complete that tutorial on the second week, therefore the deadline for this lab will be specially long **February 7 11:59 pm, (it is the same as the second lab)** 

### More information on

http://www.ee.surrey.ac.uk/Teaching/Unix/ 




