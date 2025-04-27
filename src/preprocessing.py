import os
import sys
from multiprocessing import cpu_count
import nipype.interfaces.fsl as fsl
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants import RegistrationSynQuick
from tqdm.auto import tqdm
import nibabel as nib
from brainextractor import BrainExtractor



def save_skullstripped(img_path, mask_path, output_path):
    """
    Save the skull-stripped image using the mask.
    """
    # Load the image and mask
    img = nib.load(img_path)
    mask = nib.load(mask_path)
    #Apply the mask to the image and save
    masked_data = img.get_fdata() * (mask.get_fdata() > 0)
    stripped_img = nib.Nifti1Image(masked_data, img.affine, img.header)
    stripped_img.to_filename(output_path)

def preproc_pipeline(in_file, out_dir, n_proc=None):
    """
    Perform the entire preprocessing pipeline for a single sMRI image.
    The pipeline consists of:
    1. N4 bias correction
    2. Registration to MNI space
    3. Skull stripping
    """
    # Set number of threads if n_proc is not specified
    if n_proc is None:
        n_proc = cpu_count()
    # Set up the n4 bias correction interface
    print("Performing N4 bias correction...")
    n4_corrected_path = os.path.join(out_dir, os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0]) + '_desc-preproc-N4.nii.gz'
    n4 = N4BiasFieldCorrection()
    n4.inputs.input_image = in_file
    n4.inputs.output_image = n4_corrected_path
    n4.inputs.dimension = 3
    n4.inputs.num_threads = n_proc
    n4.run()
    # Set up AntsRegistrationQuick interface
    print("Registering to MNI152 space...")
    # Redirect the standard output to the file
    reg = RegistrationSynQuick()
    reg.inputs.fixed_image = "template/mni_icbm152_t1_tal_nlin_sym_09c.nii"
    reg.inputs.moving_image = n4_corrected_path
    reg.inputs.num_threads = n_proc
    reg.inputs.output_prefix = os.path.join(out_dir, os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0] + '_space-MNI152_desc-preproc-N4')
    reg.run()
    # Rename the registered image if it exists
    registered_image_path = reg.inputs.output_prefix + 'Warped.nii.gz'
    if os.path.isfile(registered_image_path):
        os.rename(registered_image_path, reg.inputs.output_prefix + '.nii.gz')
        registered_image_path = reg.inputs.output_prefix + '.nii.gz'
    # Set up the skull stripping interface
    print("Performing skull stripping...")
    skull_mask = os.path.join(out_dir, os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0] + 'skullstrip_mask.nii.gz')
    skull_stripped = os.path.join(out_dir, os.path.splitext(os.path.splitext(os.path.basename(in_file))[0])[0] + '_space-MNI152_desc-preproc-N4-skullstripped.nii.gz')
    
    # Load the input image
    input_img = nib.load(registered_image_path)
    # Initialize the BrainExtractor with the desired fractional intensity threshold
    bet = BrainExtractor(img=input_img)
    # Run the brain extraction
    bet.run()
    # Save the computed brain mask to the output file
    bet.save_mask(skull_mask)
    save_skullstripped(registered_image_path, skull_mask, skull_stripped)

    for filename in os.listdir(out_dir):
        file_path = os.path.join(out_dir, filename)
        if os.path.isfile(
                file_path) and file_path != in_file and file_path != registered_image_path and file_path != n4_corrected_path and file_path != skull_stripped:
            os.remove(file_path)


def process_sMRIs(sMRI_files, checkpoint_file, n_proc=None):
    """
    Process a list of sMRI images and save the status of each image to a checkpoint file.
    If the checkpoint file already exists, the function will skip the images that have already been processed.
    """
    processed_sMRIs = set()
    # Check if checkpoint file exists
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_sMRIs = set(f.read().splitlines())
    for sMRI_file in tqdm(sMRI_files):
        sMRI_path = sMRI_file.path
        # Check if the sMRI has already been processed
        if sMRI_path in processed_sMRIs:
            print(f"Skipping sMRI at path: {sMRI_path}")
        else:
            # Process the sMRI
            try:
                in_file = sMRI_path
                out_dir = os.path.dirname(sMRI_path)
                preproc_pipeline(in_file=in_file, out_dir=out_dir, n_proc=n_proc)
                processed_sMRIs.add(sMRI_path)
                print(f"Processed sMRI: {sMRI_path}")
            except Exception as e:
                print(f"Error processing sMRI: {sMRI_path}")
                print(f"Error message: {str(e)}")
            # Write processed sMRI to checkpoint file
            with open(checkpoint_file, 'a') as f:
                f.write(sMRI_path + '\n')
    print("All sMRIs processed.")

