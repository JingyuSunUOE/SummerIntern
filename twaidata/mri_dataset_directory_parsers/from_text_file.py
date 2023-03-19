from twaidata.mri_dataset_directory_parsers.generic import DirectoryParser

class FromFileParser(DirectoryParser):
    def __init__(self, csv_filepath, dataset_root_in, dataset_root_out):
        """
        dataset root in: the root directory of your dataset input.
        dataset root out: the root directory of your preprocessed output folder.
        text_file: the file containing the input output filename pairs.
        the text file should be csv formatted, with each row containing
        <input file path>,<output file folder>,<output_id>_<image mode>,is_label
        
        e.g this would be a valid row for the WMH challenge dataset:

        50,Singapore/50/orig/FLAIR.nii.gz,Singapore/imgs/,FLAIR,False
        """
        
        if csv_filepath == None:
            raise ValueError("No CSV file provided!")
        
        self.csv_file = csv_filepath
        
        super().__init__(dataset_root_in, dataset_root_out)
        
    def build_dataset_table():
        with open(self.csv_filepath) as csvfile:
            reader = csvfile.reader(csvfile)
            for line in reader:
                self.files_map[line[0]] = {
                    "infile":line[1],
                    "outpath":line[2],
                    "outfilename":line[3],
                    "islabel": True if line[4].lower() == "true" else False
                }
        