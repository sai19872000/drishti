# Firestore TTL Setup

To enable the TTL policy for the `analyses` collection in Firestore, run the following one-time command:

```bash
gcloud firestore fields ttls update expire_at --collection-group=analyses --enable-ttl --project=auracle-prod-311
```
