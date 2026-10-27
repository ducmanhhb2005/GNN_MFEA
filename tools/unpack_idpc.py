"""Unpack official Mendeley IDPC-DU dataset: ZIP containing two RAR5 archives.

Can use installed libarchive (Linux), or rarfile with a configured unrar/7z.
Does not require pandas or network. Names are sanitized against path traversal.

Examples:
    python tools/unpack_idpc.py /path/to/t726xwcjf9-3.zip --set set1
    python tools/unpack_idpc.py /path/to/t726xwcjf9-3.zip --set all
"""
from __future__ import annotations
import argparse
import ctypes
import ctypes.util
from pathlib import Path
import shutil
import os
import subprocess
import tempfile
import zipfile


def _safe_target(name: str, base: Path) -> Path:
    # RAR members are set1/foo.idpc or set2/foo.idpc.
    member = Path(name.replace('\\','/'))
    if member.is_absolute() or '..' in member.parts or len(member.parts)!=2 or member.parts[0] not in {'set1','set2'} or member.suffix != '.idpc':
        raise ValueError(f'Unexpected archive member: {name!r}')
    return base / member


def _extract_unrar(rar_path: Path, destination: Path) -> int:
    """Use installed WinRAR/UnRAR on Windows without a Python dependency.

    List and validate every output name, then stream members separately so the
    archive never controls a filesystem extraction path.
    """
    executable = shutil.which("unrar")
    if not executable:
        candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"WinRAR"/"UnRAR.exe"
        if candidate.is_file():
            executable = str(candidate)
    if not executable:
        raise RuntimeError("No UnRAR executable available")
    listing = subprocess.run([executable, "lb", "-c-", str(rar_path)],
                             capture_output=True, check=True, text=True)
    members = [name.strip() for name in listing.stdout.splitlines() if name.strip().endswith(".idpc")]
    targets = [(name, _safe_target(name, destination)) for name in members]
    if not targets:
        raise RuntimeError("No IDPC members in archive")
    for name, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as stream:
            subprocess.run([executable, "p", "-inul", str(rar_path), name],
                           stdout=stream, stderr=subprocess.PIPE, check=True)
    return len(targets)


def _extract_libarchive(rar_path: Path, destination: Path) -> int:
    libname = ctypes.util.find_library('archive')
    if not libname:
        raise RuntimeError('No libarchive library available')
    la=ctypes.CDLL(libname)
    cptr=ctypes.c_void_p
    la.archive_read_new.restype=cptr
    for name in ('archive_read_support_filter_all','archive_read_support_format_all','archive_read_free'):
        getattr(la,name).argtypes=[cptr]
    la.archive_read_open_filename.argtypes=[cptr,ctypes.c_char_p,ctypes.c_size_t]
    la.archive_read_next_header.argtypes=[cptr,ctypes.POINTER(cptr)]
    la.archive_entry_pathname.argtypes=[cptr]
    la.archive_entry_pathname.restype=ctypes.c_char_p
    la.archive_read_data.argtypes=[cptr,cptr,ctypes.c_size_t]
    la.archive_read_data.restype=ctypes.c_ssize_t
    la.archive_error_string.argtypes=[cptr]
    la.archive_error_string.restype=ctypes.c_char_p
    r=la.archive_read_new(); count=0
    if not r:raise RuntimeError('Could not allocate libarchive reader')
    try:
        la.archive_read_support_filter_all(r)
        la.archive_read_support_format_all(r)
        if la.archive_read_open_filename(r,str(rar_path).encode(),65536)!=0:
            raise RuntimeError(f'Could not open {rar_path}')
        h=cptr()
        buf=ctypes.create_string_buffer(1024*1024)
        while True:
            status=la.archive_read_next_header(r,ctypes.byref(h))
            if status==1:break  # ARCHIVE_EOF
            if status<0:
                msg=la.archive_error_string(r)
                raise RuntimeError(f'RAR extraction failed: {msg!r}')
            name=la.archive_entry_pathname(h).decode('utf-8',errors='replace')
            if name.endswith('/'):
                continue
            if not name.endswith('.idpc'):
                # harmless notebook checkpoints and misc
                continue
            out=_safe_target(name,destination)
            out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('wb') as dst:
                while True:
                    count_read=la.archive_read_data(r,buf,len(buf))
                    if count_read==0:break
                    if count_read<0:
                        msg=la.archive_error_string(r)
                        raise RuntimeError(f'Error reading {name!r}: {msg!r}')
                    dst.write(buf.raw[:count_read])
            count+=1
    finally:
        la.archive_read_free(r)
    return count


def unpack(zip_path: str | Path, destination: str | Path, chosen_set: str='set1') -> dict[str,int]:
    zip_path=Path(zip_path)
    destination=Path(destination)
    if chosen_set not in ('set1','set2','all'):
        raise ValueError(chosen_set)
    sets=['set1','set2'] if chosen_set=='all' else [chosen_set]
    counts={}
    with zipfile.ZipFile(zip_path) as source:
        with tempfile.TemporaryDirectory(prefix='idpc_rar_') as temp:
            for subset in sets:
                archive=f'IDPC-DU/{subset}.rar'
                if archive not in source.namelist():
                    raise ValueError(f'Archive missing {archive}')
                rar_file=Path(temp)/f'{subset}.rar'
                with source.open(archive) as inf,rar_file.open('wb') as outf:
                    shutil.copyfileobj(inf,outf,1024*1024)
                try:
                    if os.name == "nt" and (shutil.which("unrar") or (Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"WinRAR"/"UnRAR.exe").is_file()):
                        counts[subset]=_extract_unrar(rar_file,destination)
                    else:
                        counts[subset]=_extract_libarchive(rar_file,destination)
                except (OSError,RuntimeError) as err:
                    try:
                        import rarfile
                        with rarfile.RarFile(rar_file) as rf:
                            n=0
                            for member in rf.infolist():
                                if not member.filename.endswith('.idpc'):continue
                                path=_safe_target(member.filename,destination)
                                path.parent.mkdir(parents=True,exist_ok=True)
                                with rf.open(member) as inf,path.open('wb') as outf:
                                    shutil.copyfileobj(inf,outf)
                                n+=1
                            counts[subset]=n
                    except Exception as inner:
                        raise RuntimeError('Cannot unpack nested RAR5 archives automatically. Install libarchive or unrar/7z, or extract set1.rar/set2.rar manually. '+str(inner)) from err
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',help='Original t726xwcjf9-3.zip')
    p.add_argument('--set',choices=['set1','set2','all'],default='set1')
    p.add_argument('--out',default='data/benchmark')
    args=p.parse_args()
    print('Extracted:',unpack(args.archive,args.out,args.set))
