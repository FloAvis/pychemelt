import numpy as np
import pytest

from pychemelt.utils.files import (
    get_sheet_names_of_xlsx,
    file_is_of_type_uncle,
    detect_file_type,
    detect_encoding,
    find_indexes_of_non_signal_conditions,
    find_repeated_words,
    remove_words_in_string,
    load_csv_file,
    load_quantstudio_txt,
    load_thermofluor_xlsx,
    load_nanoDSF_xlsx,
    load_panta_xlsx,
    load_uncle_multi_channel,
    load_mx3005p_txt,
    load_supr_dsf
)

nDSF_file = "./test_files/nDSFdemoFile.xlsx"
uncle_file = "./test_files/UNCLE_multi_channel.xlsx"
supr_file = "./test_files/example_data.supr"
panta_file = "./test_files/panta.xlsx"
panta_file_2 = "./test_files/panta_format_2.xlsx"

qPCR_file = "./test_files/qPCRdemoFile.xls"
quantStudio_file = "./test_files/quantStudio.txt"
MX3005P_file = "./test_files/MX3005P.txt"
csv_file = "./test_files/melting-scan.csv"
csv_file_2 = "./test_files/melting-scan_format_2.csv"

def test_get_sheet_names_of_xlsx():
    sheet_names = get_sheet_names_of_xlsx(nDSF_file)

    expected =     [
       'Overview',
       'Ratio',
       'Ratio (First Derivative)',
       '330nm',
       '330nm (First Derivative)',
       '350nm',
       '350nm (First Derivative)',
       'Scattering',
       'Scattering (First Derivative)',
    ]

    assert sheet_names == expected

def test_file_is_of_type_uncle():

    assert file_is_of_type_uncle(uncle_file)

def test_file_is_of_type_uncle_false():

    assert not file_is_of_type_uncle(MX3005P_file)

def test_detect_file_type():

    assert detect_file_type(uncle_file) == 'uncle'
    assert detect_file_type(MX3005P_file) == 'mx3005p'
    assert detect_file_type(quantStudio_file) == 'quantstudio'
    assert detect_file_type(qPCR_file) == 'thermofluor'
    assert detect_file_type(supr_file) == 'supr'
    assert detect_file_type(panta_file) == 'panta'
    assert detect_file_type(nDSF_file) == 'prometheus'
    assert detect_file_type('file.csv') == 'csv'

def test_error_on_unknown_file_type():

    with pytest.raises(ValueError):
        detect_file_type('./test_files/empty_file.noformat')

def test_detect_encoding():
    assert detect_encoding(MX3005P_file) == 'utf-8'

def test_load_two_cols_csv():

    signal_data_dic, temp_data_dic, conditions, signals = load_csv_file('./test_files/two_cols.csv')

    assert len(conditions) == 1

def test_load_many_cols_csv():

    signal_data_dic, temp_data_dic, conditions, signals = load_csv_file('./test_files/many_cols.csv')

    assert len(conditions) > 1

def test_load_csv_file_format_2():

    signal_data_dic, temp_data_dic, conditions, signals = load_csv_file(csv_file_2)

    assert len(conditions) == 1

def test_find_indexes_of_non_signal_conditions():

    signal_data = np.array([[1,2,3],[4,5,6]])
    conditions = ['signal1','signal2','Derivative']

    idx = find_indexes_of_non_signal_conditions(signal_data,conditions)

    assert idx == [2]

def test_find_repeated_words():

    strings = ['signal 1','signal 2','signal 3']
    repeated_words = find_repeated_words(strings)

    assert repeated_words == ['signal']

def test_remove_words_in_string():
    string = 'signal 1 signal 2 signal 3'
    words_to_remove = ['signal']
    output_string = remove_words_in_string(string,words_to_remove)

    assert output_string == '1 2 3'

def test_load_csv_file():

    signal_data_dic, temp_data_dic, conditions, signals = load_csv_file(csv_file)

    assert signals[0] == '350 nm'
    assert conditions[0] == 'Cap.1'
    assert len(conditions) == 47
    assert len(signal_data_dic['350 nm']) == 47
    assert len(temp_data_dic['350 nm']) == 47

def test_load_quantstudio_txt():

    signal_data_dic, temp_data_dic, conditions, signals = load_quantstudio_txt(quantStudio_file)

    assert signals[0] == 'Fluorescence'
    assert conditions[0] == 'A10'
    assert len(conditions) == 24
    assert len(signal_data_dic['Fluorescence']) == 24
    assert len(temp_data_dic['Fluorescence']) == 24

def test_load_thermofluor_xlsx():

    signal_data_dic, temp_data_dic, conditions, signals = load_thermofluor_xlsx(qPCR_file)

    assert signals[0] == 'DSF_RFU'
    assert conditions[0] == 'A01'
    assert len(conditions) == 96
    assert len(signal_data_dic['DSF_RFU']) == 96
    assert len(temp_data_dic['DSF_RFU']) == 96

def test_load_nanoDSF_xlsx():

    signal_data_dic, temp_data_dic, conditions, signals = load_nanoDSF_xlsx(nDSF_file)

    assert signals[0] == 'Ratio'
    assert conditions[0] == 'A1 GuHCl 0.05 M'
    assert len(conditions) == 48
    assert len(signal_data_dic['350nm']) == 48
    assert len(temp_data_dic['350nm']) == 48

def test_load_pant_xlsx_format_2():

    signal_data_dic, temp_data_dic, conditions, signals = load_panta_xlsx(panta_file_2)

    assert signals[0] == 'Ratio'

def test_load_panta_xlsx():

    signal_data_dic, temp_data_dic, conditions, signals = load_panta_xlsx(panta_file)

    assert signals[0] == '350 nm'
    assert conditions[0] == 'MP 1'
    assert len(conditions) == 8
    assert len(signal_data_dic['350 nm']) == 8
    assert len(temp_data_dic['350 nm']) == 8

def test_load_uncle_multi_channel():

    signal_data_dic, temp_data_dic, conditions, signals = load_uncle_multi_channel(uncle_file)

    assert signals[0] == '311.9 nm'
    assert conditions[0] == '1 mg/ml Protein'
    assert len(conditions) == 1
    assert len(signal_data_dic['311.9 nm']) == 1
    assert len(temp_data_dic['311.9 nm']) == 1

def test_load_mx3005p_txt():

    signal_data_dic, temp_data_dic, conditions, signals = load_mx3005p_txt(MX3005P_file)

    assert signals[0] == 'Fluorescence'
    assert conditions[0] == '1'
    assert len(conditions) == 3
    assert len(signal_data_dic['Fluorescence']) == 3
    assert len(temp_data_dic['Fluorescence']) == 3

def test_load_supr_dsf():

    signal_data_dic, temp_data_dic, conditions, signals = load_supr_dsf(supr_file)

    assert signals[0] == '310.5 nm'
    assert conditions[0] == 'Lyso1'
    assert len(conditions) == 384
    assert len(signal_data_dic['310.5 nm']) == 384
    assert len(temp_data_dic['310.5 nm']) == 384

