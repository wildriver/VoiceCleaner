#!/bin/bash
# Signs the app with a Developer ID certificate and notarizes it (CI only).
# Required environment (GitHub secrets):
#   MACOS_CERTIFICATE_P12       base64 of the "Developer ID Application" .p12
#   MACOS_CERTIFICATE_PASSWORD  password of the .p12
#   APPLE_API_KEY_P8            contents of the App Store Connect API key (.p8)
#   APPLE_API_KEY_ID, APPLE_API_ISSUER_ID
# usage: sign_and_notarize.sh <App.app> | --dmg <file.dmg>
set -euo pipefail

KEYCHAIN="$RUNNER_TEMP/signing.keychain-db"
KEY_FILE="$RUNNER_TEMP/AuthKey.p8"

submit() {
  printf '%s' "$APPLE_API_KEY_P8" > "$KEY_FILE"
  xcrun notarytool submit "$1" --key "$KEY_FILE" --key-id "$APPLE_API_KEY_ID" \
    --issuer "$APPLE_API_ISSUER_ID" --wait --timeout 30m
}

if [ "${1:-}" = "--dmg" ]; then
  IDENTITY=$(security find-identity -v -p codesigning "$KEYCHAIN" | grep "Developer ID Application" | head -1 | awk '{print $2}')
  codesign --force --sign "$IDENTITY" --timestamp "$2"
  submit "$2"
  xcrun stapler staple "$2"
  exit 0
fi

APP="$1"
echo "$MACOS_CERTIFICATE_P12" | base64 --decode > "$RUNNER_TEMP/certificate.p12"
security create-keychain -p ci "$KEYCHAIN"
security set-keychain-settings -lut 21600 "$KEYCHAIN"
security unlock-keychain -p ci "$KEYCHAIN"
security import "$RUNNER_TEMP/certificate.p12" -P "$MACOS_CERTIFICATE_PASSWORD" -A -t cert -f pkcs12 -k "$KEYCHAIN"
security list-keychain -d user -s "$KEYCHAIN" $(security list-keychains -d user | tr -d '"')
security set-key-partition-list -S apple-tool:,apple: -s -k ci "$KEYCHAIN" >/dev/null
IDENTITY=$(security find-identity -v -p codesigning "$KEYCHAIN" | grep "Developer ID Application" | head -1 | awk '{print $2}')
test -n "$IDENTITY"

# Sign nested code first, then the app, with the hardened runtime.
find "$APP/Contents" \( -name "*.dylib" -o -name "*.framework" -o -name "*.bundle" \) -print0 |
  while IFS= read -r -d '' item; do codesign --force --options runtime --timestamp --sign "$IDENTITY" "$item"; done
codesign --force --options runtime --timestamp --sign "$IDENTITY" "$APP"
codesign --verify --strict --verbose=2 "$APP"

if [ -n "${APPLE_API_KEY_ID:-}" ]; then
  ditto -c -k --keepParent "$APP" "$RUNNER_TEMP/app.zip"
  submit "$RUNNER_TEMP/app.zip"
  xcrun stapler staple "$APP"
fi
