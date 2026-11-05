//! Minimal AWS Signature Version 4 for S3 requests with an unsigned payload
//! (GET, HEAD, PUT, POST, with query strings): enough to talk to RustFS without
//! pulling in the AWS SDK, and leaving every request and response header in
//! our hands.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use url::Url;

type HmacSha256 = Hmac<Sha256>;

pub struct Credentials {
    pub access_key: String,
    pub secret_key: String,
    pub region: String,
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn uri_encode(input: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b'/' if !encode_slash => out.push('/'),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// The canonical query string: every pair URI-encoded, sorted by name then value.
fn canonical_query(url: &Url) -> String {
    let mut pairs: Vec<(String, String)> = url
        .query_pairs()
        .map(|(name, value)| (uri_encode(&name, true), uri_encode(&value, true)))
        .collect();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

/// Signs `method` on `url` (path-style, with its query string) and returns the
/// headers to send: the signed ones plus `extra`, which stay unsigned so the
/// server may ignore or reject them without breaking the signature.
pub fn sign(
    creds: &Credentials,
    method: &str,
    url: &Url,
    extra: &[(String, String)],
    now: time::OffsetDateTime,
) -> Vec<(String, String)> {
    let amz_date = now
        .format(&time::macros::format_description!(
            "[year][month][day]T[hour][minute][second]Z"
        ))
        .expect("format date");
    let date = &amz_date[..8];
    let host = match url.port() {
        Some(port) => format!("{}:{port}", url.host_str().unwrap_or_default()),
        None => url.host_str().unwrap_or_default().to_owned(),
    };
    let payload_hash = "UNSIGNED-PAYLOAD";
    let canonical_uri = uri_encode(url.path(), false);
    let canonical_headers =
        format!("host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n");
    let signed_headers = "host;x-amz-content-sha256;x-amz-date";
    let canonical_query = canonical_query(url);
    let canonical_request =
        format!("{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}");
    let scope = format!("{date}/{}/s3/aws4_request", creds.region);
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
        hex::encode(Sha256::digest(canonical_request.as_bytes()))
    );
    let k_date = hmac(
        format!("AWS4{}", creds.secret_key).as_bytes(),
        date.as_bytes(),
    );
    let k_region = hmac(&k_date, creds.region.as_bytes());
    let k_service = hmac(&k_region, b"s3");
    let k_signing = hmac(&k_service, b"aws4_request");
    let signature = hex::encode(hmac(&k_signing, string_to_sign.as_bytes()));
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed_headers}, Signature={signature}",
        creds.access_key
    );
    let mut headers = vec![
        ("host".to_owned(), host),
        ("x-amz-content-sha256".to_owned(), payload_hash.to_owned()),
        ("x-amz-date".to_owned(), amz_date),
        ("authorization".to_owned(), authorization),
    ];
    headers.extend(extra.iter().cloned());
    headers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_query_sorts_and_encodes() {
        let url = Url::parse("http://h:9000/b/k?uploadId=a%2Fb%3D&partNumber=2").expect("url");
        assert_eq!(canonical_query(&url), "partNumber=2&uploadId=a%2Fb%3D");
        let url = Url::parse("http://h:9000/b/k?uploads").expect("url");
        assert_eq!(canonical_query(&url), "uploads=");
        let url = Url::parse("http://h:9000/b/k").expect("url");
        assert_eq!(canonical_query(&url), "");
    }
}
