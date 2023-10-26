import os
import sys
import niconnect
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pingouin
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

plt.style.use('ggplot')
sys.path.append(os.path.join('.', 'libs'))   # add auxiliary libraries

import mitools as mi


INPUT_CONFIGURATION_FILE = os.path.join('config', 's1_extract_suvr.ini')
VALID_NORMALIZATION_METHODS = {
    'whole_cerebellum': mi.roi.ReferenceROI.aal_cerebellum
}


def calculateSUVR(_dir: str, norm_roi: list, norm_agg: str, suvr_agg: str) -> pd.DataFrame:
    """ Subrutine used to calculate the SUVR values using the whole cerebellum
    as normalization ROI. """
    assert os.path.exists(_dir)
    assert os.path.isdir(_dir)

    # load images in the input directory
    img_files = [os.path.join(_dir, f) for f in os.listdir(_dir) if f.endswith('.nii')]
    imgs = [
        mi.NiftiContainer(f) for f in tqdm(img_files, desc='loading images...')
    ]

    # extract average cerebellum metabolism
    ref_met = mi.roi.extractMetaROI(
        images=imgs,
        atlas='aal',
        rois=norm_roi,
        aggregate=norm_agg,
        n_jobs=-1,
        resample_to_atlas=True,
        verbose=True
    )
    ref_met.columns = ['normROI']

    # extract roi extractROIs
    aal_met = mi.roi.extractROIs(
        images=imgs,
        atlas='aal',
        aggregate=suvr_agg,
        n_jobs=-1,
        resample_to_atlas=True,
        verbose=True
    )

    # calculate SUVR values
    suvr_values = aal_met / ref_met.values
    suvr_values['normROI'] = ref_met.values

    # format dataframe
    suvr_values.index.names = ['img_id']
    suvr_values.index = [os.path.split(f)[-1].replace('.nii', '') for f in img_files]

    return suvr_values


if __name__ == '__main__':
    # check if input arguments have been provided
    if len(sys.argv) > 1:
        if len(sys.argv) != 2:
            raise TypeError('Only one configuration file is accepted as input')
        input_config_file = sys.argv[1]
        niconnect.io.pprint('Using user-defined configuration "%s"' % input_config_file, color='green')
    else:
        input_config_file = INPUT_CONFIGURATION_FILE   # default configuration file
        niconnect.io.pprint('Using default configuration "%s"' % input_config_file, color='green')

    # read configuration file
    config = niconnect.io.INIReader.parseFile(
        input_config_file,
        required_sections=['IN.DATA', 'OUT.DATA', 'CONFIG', 'REPORT']
    )

    # read configuration parameters
    output_dir = config['OUT.DATA']['directory']
    file_key = config['OUT.DATA']['file_key']
    output_format = config['OUT.DATA']['format']
    suvr_normalization = config['CONFIG']['suvr_normalization']
    suvr_normalization_agg = config['CONFIG']['suvr_normalization_agg']
    suvr_agg = config['CONFIG']['suvr_agg']
    report_export_distributions = config['REPORT']['export_distributions']

    assert report_export_distributions in ['true', 'false'], \
        '[REPORT] export_distributions valid values are [true, false]'

    report_export_distributions = report_export_distributions == 'true'

    if suvr_normalization not in VALID_NORMALIZATION_METHODS.keys():
        raise TypeError('Normalization ROI "%s" not defined. Available options are: %r' % (
            suvr_normalization, list(VALID_NORMALIZATION_METHODS.keys())
        ))

    # show input parameters
    print()
    niconnect.io.pprint('Output directory: "%s"' % output_dir, color='green')
    niconnect.io.pprint('SUVR normalization reference region: "%s"' % suvr_normalization, color='green')
    niconnect.io.pprint('SUVR normalization reference region aggregation: "%s"' % suvr_normalization_agg, color='green')
    niconnect.io.pprint('SUVR aggregation: "%s"' % suvr_agg, color='green')
    print()

    # extract SUVR values
    suvr_values = []
    for key, directory in config['IN.DATA'].items():
        assert os.path.exists(directory), 'Directory "%s" defined in "%s" not found' % (
            directory, input_config_file)
        assert os.path.isdir(directory), '"%s" defined in "%s" is not a directory' % (
            directory, input_config_file)
        niconnect.io.pprint('Processing: "%s"' % key, color='green')
        suvr = calculateSUVR(
            directory,
            norm_roi=VALID_NORMALIZATION_METHODS[suvr_normalization],
            norm_agg=suvr_normalization_agg,
            suvr_agg=suvr_agg
        )
        suvr['key'] = key
        suvr_values.append(suvr)
    suvr_values_df = pd.concat(suvr_values, axis=0)

    # create the output directory if not exists
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True)
        niconnect.io.pprint('\nCreated output directory: "%s"' % output_dir, color='green')

    # export the generated data
    curr_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, '%s_%s.%s' % (curr_time, file_key, output_format))
    if output_format == 'parquet':
        suvr_values_df.to_parquet(output_file)
    elif output_format == 'csv':
        suvr_values_df.to_csv(output_file)
    else:
        raise TypeError('Invalid output format "%s"' % output_format)

    # create a report subdirectory
    report_directory = os.path.join(output_dir, '%s_%s_report' % (curr_time, file_key))
    Path(report_directory).mkdir(parents=True)

    # ======== perform a PCA to analyze the resulting data
    # select the ROIs to analyze
    rois = [c for c in suvr_values_df.columns if c not in ['normROI', 'key']]
    # get cohorts
    unique_cohorts = list(config['IN.DATA'].keys())
    # fit PCA
    pca = PCA(n_components=2).fit(suvr_values_df[rois])
    # calculate PCs
    PCs = pca.transform(suvr_values_df[rois])

    cmap = plt.get_cmap('viridis', len(unique_cohorts))
    fig, ax = plt.subplots()
    fig.set_dpi(150)

    for i, cohort in enumerate(unique_cohorts):
        cohort_mask = (suvr_values_df['key'] == cohort).values
        ax.scatter(
            PCs[cohort_mask, 0],
            PCs[cohort_mask, 1],
            c=[cmap(i)] * np.sum(cohort_mask),
            label=cohort)
    ax.legend()
    ax.set_title('Explained variance: {:.2f}%'.format(pca.explained_variance_.cumsum()[-1] * 100))
    ax.set_ylabel('PC2')
    ax.set_ylabel('PC1')
    plt.savefig(os.path.join(report_directory, 'PCA.png'), dpi=300)

    # ======== perform an ANOVA ROI-level comparison to analyze the resulting data
    anova_df = []
    for roi in rois:
        anova_ = pingouin.anova(
            data=suvr_values_df,
            dv=roi,
            between='key'
        )
        anova_['roi'] = roi
        anova_df.append(anova_)

    anova_df = pd.concat(anova_df, axis=0)
    anova_df = anova_df.sort_values(by='p-unc', ascending=False)
    anova_df = anova_df.drop(columns=['Source', 'ddof1', 'ddof2']).set_index('roi')
    anova_df['p-val-fdr05'] = multipletests(
        pvals=anova_df['p-unc'].values,
        method='fdr_bh'
    )[1]
    anova_df.to_excel(os.path.join(report_directory, 'ANOVA.xlsx'))

    # ======= Display ROI-level distributions
    if report_export_distributions:
        Path(os.path.join(report_directory, 'distributions')).mkdir(parents=True)

        # plot ROI distributions
        for i in range(4, len(rois), 4):
            fig, axes = plt.subplots(1, 4, figsize=(20, 3.5))
            fig.set_dpi(150)

            for ii, roi in enumerate(rois[i - 4:i]):
                ax = sns.histplot(
                    data=suvr_values_df,
                    x=roi,
                    hue='key',
                    common_norm=False,
                    stat='density',
                    kde=True,
                    ax=axes[ii]
                )
                sns.move_legend(
                    ax, "lower center",
                    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)

            plt.savefig(
                os.path.join(report_directory, 'distributions', '{}.png'.format('-'.join(rois[i - 4:i]))),
                dpi=300, bbox_inches='tight')
            plt.close()
