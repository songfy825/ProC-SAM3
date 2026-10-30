import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


class Assets(unittest.TestCase):
    def entry(self,root):
        p=root/'source.bin';p.write_bytes(b'actual fixture bytes'*50)
        return {'source':str(p),'name':'weights/sam3.pt','bytes':p.stat().st_size,
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

    def test_archive_roundtrip_hashes_every_file(self):
        from scripts.asset_transfer.package import pack_files,verify_archive
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);entry=self.entry(root);archive=root/'bundle.tar.gz'
            pack_files([entry],archive,{'scope':'synthetic packing fixture'})
            result=verify_archive(archive)
            self.assertEqual(result['files'],1)
            self.assertEqual(result['payload_bytes'],entry['bytes'])
            with tarfile.open(archive,'r:gz') as tf:
                self.assertEqual(tf.extractfile('p01-assets/weights/sam3.pt').read(),Path(entry['source']).read_bytes())

    def test_changed_source_fails_without_publishing_archive(self):
        from scripts.asset_transfer.package import pack_files
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);entry=self.entry(root);entry['sha256']='0'*64
            with self.assertRaises(ValueError):pack_files([entry],root/'bad.tar.gz',{})
            self.assertFalse((root/'bad.tar.gz').exists())

    def test_valid_gzip_with_corrupted_member_is_rejected(self):
        from scripts.asset_transfer.package import pack_files,verify_archive
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);entry=self.entry(root);archive=root/'good.tar.gz';bad=root/'bad.tar.gz'
            pack_files([entry],archive,{})
            with tarfile.open(archive,'r:gz') as source,tarfile.open(bad,'w:gz') as target:
                for member in source:
                    data=source.extractfile(member).read() if member.isfile() else None
                    if member.name.endswith('sam3.pt'):data=b'X'+data[1:]
                    target.addfile(member,io.BytesIO(data) if data is not None else None)
            with self.assertRaises(ValueError):verify_archive(bad)

    def test_unsafe_archive_path_is_rejected(self):
        from scripts.asset_transfer.package import pack_files
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);entry=self.entry(root);entry['name']='../escape'
            with self.assertRaises(ValueError):pack_files([entry],root/'unsafe.tar.gz',{})

    def test_missing_gzip_footer_is_rejected(self):
        from scripts.asset_transfer.package import pack_files,verify_archive
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);archive=root/'good.tar.gz';pack_files([self.entry(root)],archive,{})
            bad=root/'truncated.tar.gz';bad.write_bytes(archive.read_bytes()[:-8])
            with self.assertRaises((ValueError,OSError,EOFError)):verify_archive(bad)

    def test_corrupt_gzip_crc_is_rejected(self):
        from scripts.asset_transfer.package import pack_files,verify_archive
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);archive=root/'good.tar.gz';pack_files([self.entry(root)],archive,{})
            data=bytearray(archive.read_bytes());data[-8]^=0xff
            bad=root/'crc.tar.gz';bad.write_bytes(data)
            with self.assertRaises((ValueError,OSError,EOFError)):verify_archive(bad)

    def fixture(self,root):
        bundle=root/'bundle'
        for sub in ('weights','datasets/LoveDA','datasets/iSAID','manifests','inputs','runtime'):
            (bundle/sub).mkdir(parents=True,exist_ok=True)
        (bundle/'weights/sam3.pt').write_bytes(b'link test only')
        return bundle

    def test_links_preserve_both_path_schemes_and_are_idempotent(self):
        from scripts.asset_transfer.setup_links import install_links
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);bundle=self.fixture(root);prefix=root/'target';prefix.mkdir()
            install_links(bundle,None,prefix,apply=True)
            for relative in ('home/fengyi/sam3.pt','data/weights/sam3.pt'):
                p=prefix/relative
                self.assertTrue(p.is_symlink());self.assertEqual(p.read_bytes(),b'link test only')
            self.assertEqual((prefix/'home/fengyi/E/data/SegEarth/iSAID').resolve(),bundle/'datasets/iSAID')
            self.assertEqual((prefix/'data/p01-inputs').resolve(),bundle/'inputs')
            install_links(bundle,None,prefix,apply=True)

    def test_conflicting_path_aborts_before_any_link_creation(self):
        from scripts.asset_transfer.setup_links import install_links
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);bundle=self.fixture(root);prefix=root/'target'
            old=prefix/'data/weights/sam3.pt';old.parent.mkdir(parents=True);old.write_bytes(b'existing user file')
            with self.assertRaises(FileExistsError):install_links(bundle,None,prefix,apply=True)
            self.assertEqual(old.read_bytes(),b'existing user file')
            self.assertFalse((prefix/'home/fengyi/sam3.pt').exists())

    def test_dry_run_creates_no_target_paths(self):
        from scripts.asset_transfer.setup_links import install_links
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);bundle=self.fixture(root);prefix=root/'target';prefix.mkdir()
            install_links(bundle,None,prefix,apply=False)
            self.assertEqual(list(prefix.iterdir()),[])

    def test_existing_parent_alias_does_not_create_same_link_twice(self):
        from scripts.asset_transfer.setup_links import install_links
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);bundle=self.fixture(root);prefix=root/'target'
            physical=prefix/'home/fengyi/E/data';physical.mkdir(parents=True)
            (prefix/'home/fengyi/data').symlink_to(physical)
            install_links(bundle,None,prefix,apply=True)
            self.assertEqual((prefix/'home/fengyi/data/SegEarth/LoveDA').resolve(),bundle/'datasets/LoveDA')

    def test_dangling_parent_aborts_before_any_link_creation(self):
        from scripts.asset_transfer.setup_links import install_links
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);bundle=self.fixture(root);prefix=root/'target';prefix.mkdir()
            (prefix/'data').symlink_to(prefix/'missing-data')
            with self.assertRaises(FileExistsError):install_links(bundle,None,prefix,apply=True)
            self.assertFalse((prefix/'home').exists())

    def test_existing_target_data_and_weight_need_no_payload_duplicates(self):
        from scripts.asset_transfer.setup_links import install_links
        with tempfile.TemporaryDirectory(dir='/tmp') as d:
            root=Path(d);bundle=root/'metadata';bundle.mkdir()
            for sub in ('manifests','inputs','runtime'):(bundle/sub).mkdir()
            data=root/'existing/SegEarth'
            for name in ('loveDA','iSAID'):(data/name).mkdir(parents=True)
            checkpoint=root/'existing/sam3.pt';checkpoint.write_bytes(b'weight fixture')
            prefix=root/'target';prefix.mkdir()
            install_links(bundle,None,prefix,True,existing_data_root=data,loveda_name='loveDA',checkpoint=checkpoint)
            self.assertEqual((prefix/'home/fengyi/E/data/SegEarth/LoveDA').resolve(),data/'loveDA')
            self.assertEqual((prefix/'data/weights/sam3.pt').read_bytes(),b'weight fixture')
            self.assertFalse((bundle/'datasets').exists())


if __name__=='__main__':unittest.main()
