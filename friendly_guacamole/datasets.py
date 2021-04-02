from typing import Dict, Callable
from tempfile import NamedTemporaryFile
from pathlib import Path
from abc import ABC
from urllib import request
from io import BytesIO
from zipfile import ZipFile
import h5py
import os
import pandas as pd
from biom import load_table
from biom.util import biom_open
from friendly_guacamole.exceptions import DatasetError
from http.client import HTTPResponse


class Dataset(ABC):
    artifacts: Dict[str, 'ArtifactInterface'] = dict()

    def __init__(self, path, download=True):
        self.path = path

        if download:
            self.download()

        if not self._check_integrity():
            raise DatasetError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self._data = {}

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        print('Downloading...', end='', flush=True)
        for artifact in self.artifacts.values():
            artifact.download(self.path)
        print('Done.')

    def _check_integrity(self):
        return all(
            artifact.exists(self.path) for artifact in
            self.artifacts.values()
        )

    def __getitem__(self, item):
        if item in self._data:
            return self._data[item]
        elif item in self.artifacts:
            value = self.artifacts[item].read(self.path)
            self._data[item] = value
            return value
        else:
            raise KeyError(item)

    def apply(self, attr, method_name):
        data = self.artifacts[attr]
        method = getattr(data, method_name)
        return method(self)


class ArtifactInterface:

    def download(self, path):
        data = self.client.download()
        self.archiver.save(data, path)
        return data

    def read(self, path):
        data = self.archiver.read(path)
        return data

    def exists(self, path):
        return self.archiver.exists(path)


class ClientInterface:

    def download(self):
        raise NotImplementedError()


class QiitaClientInterface(ClientInterface):
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    @staticmethod
    def make_request(url):
        return request.urlopen(url)

    def get_data(self):
        link = self.get_link()
        zipdata = BytesIO()
        zipdata.write(self.make_request(link).read())
        myzipfile = ZipFile(zipdata)
        return myzipfile

    def download(self):
        data = self.get_data()
        return self.scavenge_data(data)


class WebClientInterface(ClientInterface):
    @staticmethod
    def make_request(url):
        return request.urlopen(url)

    def get_data(self):
        link = self.get_link()
        data = self.make_request(link).read()
        return data

    def download(self):
        data = self.get_data()
        return self.scavenge_data(data)


class PassthroughWebClient(WebClientInterface):
    def __init__(self, link):
        self.link = link

    def get_link(self):
        return self.link

    def scavenge_data(self, data):
        return data


class QiitaArtifactClientInterface(QiitaClientInterface):
    _artifact_fstring = "https://qiita.ucsd.edu/public_artifact_download/" \
                        "?artifact_id={}"

    def __init__(self, artifact_id):
        self.artifact_id = artifact_id

    def get_link(self):
        artifact_link = self._artifact_fstring.format(self.artifact_id)
        return artifact_link


class QiitaMetadataClient(QiitaClientInterface):
    _metadata_fstring = "https://qiita.ucsd.edu/public_download/" \
                        "?data=sample_information&study_id={}"

    def __init__(self, study_id):
        self.study_id = study_id

    def get_link(self):
        artifact_link = self._metadata_fstring.format(self.study_id)
        return artifact_link

    def scavenge_data(self, data):
        path_to_metadata = 'templates'
        md_candidates = list(
            filter(
                lambda n: n.startswith(path_to_metadata) and n.endswith('.txt'),
                data.namelist()
            ))
        if len(md_candidates) == 0:
            return False
        else:
            md_file_path = md_candidates[0]
        df = pd.read_csv(data.open(md_file_path), sep="\t")
        return df

    @classmethod
    def merge(cls, list_of_metadata):
        return pd.concat(list_of_metadata)


class QiitaTableClient(QiitaArtifactClientInterface):

    def scavenge_data(self, data):
        path_to_table = os.path.join(
            'BIOM',
            str(self.artifact_id),
            'otu_table.biom',
        )
        return load_table(h5py.File(data.open(path_to_table)))

    @classmethod
    def merge(cls, list_of_data):
        if len(list_of_data) > 0:
            data = list_of_data[0]
            merged = data
        else:
            raise DatasetError('No data retrieved.')
        for data in list_of_data[1:]:
            merged = merged.merge(data)
        return merged


class ClientList(ClientInterface):

    def __init__(self, clients):
        self.clients = clients

    def download(self):
        first_client = self.clients[0]
        downloads = [client.download() for client in self.clients]
        return type(first_client).merge(downloads)


class ArchiverInterface:

    def save(self, data, path):
        raise NotImplementedError()

    def read(self, path):
        raise NotImplementedError()

    def exists(self, path):
        raise NotImplementedError()


class BIOMArchiver(ArchiverInterface):

    def __init__(self, identifier):
        self.identifier = str(identifier)

    def path(self, root):
        pardir = os.path.join(root, 'table', self.identifier)
        path = os.path.join(pardir, self.identifier +
                            '.table.biom')
        return path

    def exists(self, root):
        return os.path.exists(self.path(root))

    def read(self, root):
        path = self.path(root)
        return load_table(path)

    def save(self, data, root):
        path = self.path(root)
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with biom_open(path, 'w') as fh:
            data.to_hdf5(fh, self.identifier)


class MetadataArchiver(ArchiverInterface):
    def __init__(self, identifier):
        self.identifier = str(identifier)

    def path(self, root):
        pardir = os.path.join(root, 'metadata', self.identifier)
        path = os.path.join(pardir, self.identifier + '.metadata.tsv')
        return path

    def exists(self, root):
        return os.path.exists(self.path(root))

    def read(self, root):
        return pd.read_csv(self.path(root), sep='\t', index_col=0)

    def save(self, data, root):
        path = self.path(root)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, sep='\t')


class NewickArchiver(ArchiverInterface):
    def __init__(self, identifier):
        self.identifier = str(identifier)

    def path(self, root):
        return os.path.join(root, 'tree', self.identifier, self.identifier +
                            '.tree.nwk')

    def exists(self, root):
        return os.path.exists(self.path(root))

    def read(self, root):
        return open(self.path(root)).read()

    def save(self, data, root):
        path = self.path(root)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        open(path, 'wb').write(data)


class QiitaMetadata(ArtifactInterface):

    def __init__(self, study_id):
        self.archiver = MetadataArchiver(study_id)
        self.client = QiitaMetadataClient(study_id)


class QiitaTable(ArtifactInterface):

    def __init__(self, artifact_id):
        self.archiver = BIOMArchiver(artifact_id)
        self.client = QiitaTableClient(artifact_id)


class TableList(ArtifactInterface):

    def __init__(self, artifact_id_list):
        self.archiver = BIOMArchiver('.'.join(str(id_) for id_ in
                                              artifact_id_list))
        self.client = ClientList([QiitaTableClient(id_)
                                  for id_ in artifact_id_list])


class MetadataList(ArtifactInterface):

    def __init__(self, study_id_list):
        self.archiver = MetadataArchiver('.'.join(str(id_) for id_ in
                                                  study_id_list))
        self.client = ClientList([QiitaMetadataClient(id_)
                                  for id_ in study_id_list])


class GreenGenes97Tree(ArtifactInterface):

    link = 'ftp://greengenes.microbio.me/greengenes_release/gg_13_8_otus/' \
           'trees/97_otus.tree'

    def __init__(self):
        self.archiver = NewickArchiver('gg_13_8_97_otus')
        self.client = PassthroughWebClient(self.link)
        self._tree_file = None

    def path(self, dataset):
        if self._tree_file is None:
            self._tree_file = NamedTemporaryFile('w')
            tree = self.archiver.read(dataset.path)
            self._tree_file.write(tree)
            self._tree_file.flush()
        return self._tree_file.name


class KeyboardDataset(Dataset):
    study_id = 232
    table_artifact_id = 46809

    artifacts = {
        'table': QiitaTable(table_artifact_id),
        'metadata': QiitaMetadata(study_id),
        'tree': GreenGenes97Tree(),
    }


class DietInterventionStudy(Dataset):
    study_id = 11550
    table_artifact_ids = [63512, 63515]

    artifacts = {
        'metadata': QiitaMetadata(study_id),
        'table': TableList(table_artifact_ids),
        'tree': GreenGenes97Tree(),
    }


class HMPV13V35(Dataset):

    study_ids = [1927, 1928]
    table_artifact_ids = [47414, 47420]

    artifacts = {
        'metadata': MetadataList(study_ids),
        'table': TableList(table_artifact_ids),
        'tree': GreenGenes97Tree(),
    }
