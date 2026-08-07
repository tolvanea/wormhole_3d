//! Opening asset files, and explaining it properly when that fails.
//!
//! This renderer is mostly assets: two cube maps, a glTF file, the buffer and
//! the dozen textures that glTF points at. All of them are named by paths
//! relative to the working directory, which means the single most likely way
//! to fail is a path that is slightly wrong or a program started from the
//! wrong directory. A bare "No such file or directory (os error 2)" is close
//! to useless for either.
//!
//! So when a path does not resolve, say: what we were trying to open, the path
//! as given, what it expanded to, how far down that path actually exists, what
//! is there instead, and -- most usefully -- whether a file of that name is
//! sitting somewhere nearby.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

/// How deep to go looking for a file with the right name once the given path
/// has turned out to be wrong, and how many directory entries to look at in
/// total. Both are just there to stop a stray `--model /` walking the disk.
const SEARCH_DEPTH: usize = 4;
const SEARCH_BUDGET: usize = 4000;

/// Read a file whole, or fail with a diagnosis. `what` names the asset in
/// human terms ("the glTF model", "the sky cube map for universe A").
pub fn read(path: &Path, what: &str) -> Result<Vec<u8>> {
    ensure_readable(path, what)?;
    std::fs::read(path).with_context(|| format!("cannot read {what}: {}", path.display()))
}

/// Decode an image, or fail with a diagnosis.
pub fn open_image(path: &Path, what: &str) -> Result<image::DynamicImage> {
    ensure_readable(path, what)?;
    // Past this point the file exists and is readable, so any failure is about
    // the *contents* -- a truncated PNG, an unsupported format -- and the
    // decoder's own message is the informative one.
    image::open(path).with_context(|| format!("cannot decode {what}: {}", path.display()))
}

/// Create a directory and everything above it, or fail with a diagnosis.
pub fn create_dir(path: &Path, what: &str) -> Result<()> {
    std::fs::create_dir_all(path).map_err(|e| {
        let mut msg = format!("cannot create {what}: {}\n\n  {e}", display_full(path));
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            msg.push_str("\n\n  check the permissions on the directory above it");
        }
        anyhow!(msg)
    })
}

/// Create (or truncate) a file for writing, or fail with a diagnosis.
pub fn create_file(path: &Path, what: &str) -> Result<std::fs::File> {
    std::fs::File::create(path).map_err(|e| {
        let mut msg = format!("cannot write {what}: {}\n\n  {e}", display_full(path));
        if let Some(dir) = path.parent()
            && !dir.as_os_str().is_empty()
            && !dir.is_dir()
        {
            msg.push_str(&format!(
                "\n\n  the directory {} does not exist",
                dir.display()
            ));
        }
        anyhow!(msg)
    })
}

// ------------------------------- diagnosis ---------------------------------

/// The check that produces the good message. Everything else here is plumbing.
fn ensure_readable(path: &Path, what: &str) -> Result<()> {
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => Err(anyhow!(
            "{}",
            report(
                what,
                path,
                "that is a directory, not a file",
                &suggest_inside(path),
            )
        )),
        Ok(_) => {
            // Exists, but we may still not be allowed to read it.
            match std::fs::File::open(path) {
                Ok(_) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => Err(anyhow!(
                    "{}",
                    report(
                        what,
                        path,
                        "permission denied",
                        &["check the file mode".to_string()],
                    )
                )),
                Err(e) => Err(anyhow!("{}", report(what, path, &e.to_string(), &[]))),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(anyhow!("{}", report(what, path, "no such file", &suggest(path))))
        }
        Err(e) => Err(anyhow!("{}", report(what, path, &e.to_string(), &[]))),
    }
}

/// Assemble the message. Kept to one shape so every asset failure reads the
/// same way whichever module it came from.
fn report(what: &str, path: &Path, problem: &str, hints: &[String]) -> String {
    let mut s = format!("cannot open {what}\n\n  {}\n\n  {problem}", display_full(path));
    for hint in hints {
        s.push_str(&format!("\n\n  {hint}"));
    }
    if path.is_relative()
        && let Ok(cwd) = std::env::current_dir()
    {
        s.push_str(&format!(
            "\n\n  paths are relative to the working directory, {}\n  \
             (run from the project root, or pass an absolute path)",
            cwd.display()
        ));
    }
    s
}

/// The path as written, plus what it actually expands to when that differs.
fn display_full(path: &Path) -> String {
    match std::path::absolute(path) {
        Ok(abs) if abs != path => format!("{}\n  -> {}", path.display(), abs.display()),
        _ => path.display().to_string(),
    }
}

/// What to say about a path that does not exist.
fn suggest(path: &Path) -> Vec<String> {
    let mut out = Vec::new();

    // How far down the path is real? That localises the mistake for the user
    // far better than "the whole thing is wrong" does.
    let deepest = deepest_existing(path);
    match &deepest {
        Some(dir) if Some(dir.as_path()) == path.parent() => {
            out.push(format!(
                "the directory {} exists but has nothing by that name in it",
                dir.display()
            ));
        }
        Some(dir) => out.push(format!(
            "the path stops being real after {}",
            dir.display()
        )),
        None => {}
    }

    // The most useful hint by far: the file they meant is probably right
    // there, one directory over. Sample models in particular nest the .gltf
    // inside a folder of the same name.
    if let Some(name) = path.file_name() {
        let root = deepest.clone().unwrap_or_else(|| PathBuf::from("."));
        let found = find_named(&root, name, SEARCH_DEPTH);
        if !found.is_empty() {
            let list = found
                .iter()
                .map(|p| format!("\n    {}", p.display()))
                .collect::<String>();
            out.push(format!("but a file of that name does exist nearby:{list}"));
            return out; // a direct hit beats listing the directory
        }
    }

    // Otherwise show what is actually there, preferring same-extension files.
    if let Some(dir) = &deepest {
        let listing = list_dir(dir, path.extension().and_then(|e| e.to_str()));
        if !listing.is_empty() {
            out.push(format!(
                "{} contains:{}",
                dir.display(),
                listing
                    .iter()
                    .map(|p| format!("\n    {p}"))
                    .collect::<String>()
            ));
        }
    }
    out
}

/// For "you gave me a directory": point at the model files inside it. Sample
/// assets nest the .gltf a couple of levels down, so this searches rather than
/// just listing -- naming the directory is the natural mistake to make.
fn suggest_inside(dir: &Path) -> Vec<String> {
    let found = find_by_extension(dir, &["gltf", "glb"], SEARCH_DEPTH);
    if found.is_empty() {
        return Vec::new();
    }
    vec![format!(
        "did you mean:{}",
        found
            .iter()
            .map(|p| format!("\n    {}", p.display()))
            .collect::<String>()
    )]
}

/// The longest prefix of `path` that exists, so the message can say where
/// reality and the path part company.
fn deepest_existing(path: &Path) -> Option<PathBuf> {
    let mut best = None;
    let mut acc = PathBuf::new();
    for part in path.components() {
        acc.push(part);
        if acc.is_dir() {
            best = Some(acc.clone());
        } else {
            break;
        }
    }
    best
}

/// Hunt for a file called `name` under `root`.
fn find_named(root: &Path, name: &std::ffi::OsStr, max_depth: usize) -> Vec<PathBuf> {
    walk(root, max_depth, |e| e.file_name() == name)
}

/// Hunt for files with any of `exts` under `root`.
fn find_by_extension(root: &Path, exts: &[&str], max_depth: usize) -> Vec<PathBuf> {
    walk(root, max_depth, |e| {
        e.path()
            .extension()
            .and_then(|x| x.to_str())
            .is_some_and(|x| exts.iter().any(|w| w.eq_ignore_ascii_case(x)))
    })
}

/// Bounded directory walk collecting files that match. This only runs on the
/// failure path, but it should still never be the reason the program hangs --
/// hence the caps on depth, on entries examined, and on hits returned.
fn walk(root: &Path, max_depth: usize, matches: impl Fn(&std::fs::DirEntry) -> bool) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut queue = vec![(root.to_path_buf(), 0usize)];
    let mut budget = SEARCH_BUDGET;

    while let Some((dir, depth)) = queue.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if budget == 0 {
                return found;
            }
            budget -= 1;
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                // Skip the places that are always large and never the answer.
                let base = entry.file_name();
                if base == "target" || base == ".git" || base == "node_modules" {
                    continue;
                }
                if depth < max_depth {
                    queue.push((entry.path(), depth + 1));
                }
            } else if matches(&entry) {
                found.push(entry.path());
                if found.len() >= 3 {
                    return found;
                }
            }
        }
    }
    found
}

/// Up to a handful of entries in `dir`, floating files with the wanted
/// extension to the top so the relevant ones survive the truncation.
fn list_dir(dir: &Path, want_ext: Option<&str>) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<(bool, bool, String)> = entries
        .flatten()
        .map(|e| {
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let path = e.path();
            let matches = want_ext.is_some_and(|w| {
                path.extension().and_then(|x| x.to_str()) == Some(w)
            });
            let name = e.file_name().to_string_lossy().into_owned();
            (!matches, !is_dir, if is_dir { format!("{name}/") } else { name })
        })
        .collect();
    names.sort();
    let total = names.len();
    let mut out: Vec<String> = names.into_iter().take(8).map(|(_, _, n)| n).collect();
    if total > out.len() {
        out.push(format!("... and {} more", total - out.len()));
    }
    out
}
