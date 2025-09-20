#!/usr/bin/env python3
"""
Apple Photos search script using OpenAI-compatible API and SQLite queries.
Converts natural language queries to SQL queries for Apple Photos database.
"""

import argparse
import re
import sqlite3
import os
import random
import subprocess
import pandas as pd
from pathlib import Path
from openai import OpenAI
import osxphotos
from osxphotos import ExportOptions, PhotoExporter, PhotosAlbum

# Example Q&A pairs for the AI model
EXAMPLE_QA_PAIRS = [
    {
        "question": "What is the table schema for ZPERSON?",
        "answer": """
```sql
CREATE TABLE ZPERSON (
    Z_PK INTEGER PRIMARY KEY,
    Z_ENT INTEGER,
    Z_OPT INTEGER,
    ZAGETYPE INTEGER,
    ZCLOUDDELETESTATE INTEGER,
    ZCLOUDLOCALSTATE INTEGER,
    ZCLOUDVERIFIEDTYPE INTEGER,
    ZDETECTIONTYPE INTEGER,
    ZFACECOUNT INTEGER,
    ZGENDERTYPE INTEGER,
    ZINPERSONNAMINGMODEL INTEGER,
    ZKEYFACEPICKSOURCE INTEGER,
    ZMANUALORDER INTEGER,
    ZQUESTIONTYPE INTEGER,
    ZSUGGESTEDFORCLIENTTYPE INTEGER,
    ZTYPE INTEGER,
    ZVERIFIEDTYPE INTEGER,
    ZASSOCIATEDFACEGROUP INTEGER,
    ZKEYFACE INTEGER,
    ZMERGETARGETPERSON INTEGER,
    ZDISPLAYNAME VARCHAR,
    ZFULLNAME VARCHAR,
    ZPERSONUUID VARCHAR,
    ZPERSONURI VARCHAR,
    ZCONTACTMATCHINGDICTIONARY BLOB,
    ZMERGECANDIDATECONFIDENCE FLOAT,
    ZSHAREPARTICIPANT INTEGER,
    ZASSETSORTORDER INTEGER,
    ZMDID VARCHAR,
    ZCLOUDDETECTIONTYPE INTEGER,
    ZISMECONFIDENCE FLOAT
)
```
""",
    },
    {
        "question": "What is the table schema for ZDETECTEDFACE?",
        "answer": """
```sql
CREATE TABLE ZDETECTEDFACE (
    Z_PK INTEGER PRIMARY KEY,
    Z_ENT INTEGER,
    Z_OPT INTEGER,
    ZAGETYPE INTEGER,
    ZASSETVISIBLE INTEGER,
    ZCLOUDLOCALSTATE INTEGER,
    ZCLOUDNAMESOURCE INTEGER,
    ZCLUSTERSEQUENCENUMBER INTEGER,
    ZCONFIRMEDFACECROPGENERATIONSTATE INTEGER,
    ZDETECTIONTYPE INTEGER,
    ZETHNICITYTYPE INTEGER,
    ZEYEMAKEUPTYPE INTEGER,
    ZEYESSTATE INTEGER,
    ZFACEALGORITHMVERSION INTEGER,
    ZFACEEXPRESSIONTYPE INTEGER,
    ZFACIALHAIRTYPE INTEGER,
    ZGAZETYPE INTEGER,
    ZGENDERTYPE INTEGER,
    ZGLASSESTYPE INTEGER,
    ZHAIRCOLORTYPE INTEGER,
    ZHAIRTYPE INTEGER,
    ZHASFACEMASK INTEGER,
    ZHASSMILE INTEGER,
    ZHEADGEARTYPE INTEGER,
    ZHIDDEN INTEGER,
    ZISINTRASH INTEGER,
    ZISLEFTEYECLOSED INTEGER,
    ZISRIGHTEYECLOSED INTEGER,
    ZLIPMAKEUPTYPE INTEGER,
    ZMANUAL INTEGER,
    ZNAMESOURCE INTEGER,
    ZPOSETYPE INTEGER,
    ZQUALITYMEASURE INTEGER,
    ZSKINTONETYPE INTEGER,
    ZSMILETYPE INTEGER,
    ZSOURCEHEIGHT INTEGER,
    ZSOURCEWIDTH INTEGER,
    ZTRAININGTYPE INTEGER,
    ZVIPMODELTYPE INTEGER,
    ZASSETFORFACE INTEGER,
    ZFACECROP INTEGER,
    ZFACEGROUP INTEGER,
    ZFACEGROUPBEINGKEYFACE INTEGER,
    ZFACEPRINT INTEGER,
    ZPERSONFORFACE INTEGER,
    ZPERSONBEINGKEYFACE INTEGER,
    ZADJUSTMENTVERSION TIMESTAMP,
    ZBLURSCORE FLOAT,
    ZBODYCENTERX FLOAT,
    ZBODYCENTERY FLOAT,
    ZBODYHEIGHT FLOAT,
    ZBODYWIDTH FLOAT,
    ZCENTERX FLOAT,
    ZCENTERY FLOAT,
    ZGAZECENTERX FLOAT,
    ZGAZECENTERY FLOAT,
    ZPOSEYAW FLOAT,
    ZQUALITY FLOAT,
    ZROLL FLOAT,
    ZSIZE FLOAT,
    ZGROUPINGIDENTIFIER VARCHAR,
    ZMASTERIDENTIFIER VARCHAR,
    ZUUID VARCHAR,
    ZVUOBSERVATIONID INTEGER,
    ZASSETFORTORSO INTEGER,
    ZASSETFORTEMPORALDETECTEDFACES INTEGER,
    ZPERSONFORTORSO INTEGER,
    ZPERSONFORTEMPORALDETECTEDFACES INTEGER,
    ZGAZECONFIDENCE FLOAT,
    ZGAZERECTSTRING VARCHAR,
    ZGAZEANGLE FLOAT,
    ZSTARTTIME FLOAT,
    ZDURATION FLOAT
)
```
""",
    },
    {
        "question": "What is the table schema for ZASSET?",
        "answer": """
```sql
CREATE TABLE ZASSET (
    Z_PK INTEGER PRIMARY KEY,
    Z_ENT INTEGER,
    Z_OPT INTEGER,
    ZACTIVELIBRARYSCOPEPARTICIPATIONSTATE INTEGER,
    ZADJUSTMENTSSTATE INTEGER,
    ZAVALANCHEKIND INTEGER,
    ZAVALANCHEPICKTYPE INTEGER,
    ZBUNDLESCOPE INTEGER,
    ZCAMERAPROCESSINGADJUSTMENTSTATE INTEGER,
    ZCLOUDDELETESTATE INTEGER,
    ZCLOUDDOWNLOADREQUESTS INTEGER,
    ZCLOUDHASCOMMENTSBYME INTEGER,
    ZCLOUDHASCOMMENTSCONVERSATION INTEGER,
    ZCLOUDHASUNSEENCOMMENTS INTEGER,
    ZCLOUDISDELETABLE INTEGER,
    ZCLOUDISMYASSET INTEGER,
    ZCLOUDLOCALSTATE INTEGER,
    ZCLOUDPLACEHOLDERKIND INTEGER,
    ZCOMPLETE INTEGER,
    ZCURRENTSLEETCAST INTEGER,
    ZDEFERREDPROCESSINGNEEDED INTEGER,
    ZDEPTHTYPE INTEGER,
    ZDERIVEDCAMERACAPTUREDEVICE INTEGER,
    ZDUPLICATEASSETVISIBILITYSTATE INTEGER,
    ZFACEAREAPOINTS INTEGER,
    ZFAVORITE INTEGER,
    ZGENERATIVEMEMORYCREATIONELIGIBILITYSTATE INTEGER,
    ZHDRTYPE INTEGER,
    ZHEIGHT INTEGER,
    ZHIDDEN INTEGER,
    ZHIGHFRAMERATESTATE INTEGER,
    ZISDETECTEDSCREENSHOT INTEGER,
    ZISMAGICCARPET INTEGER,
    ZISRECENTLYSAVED INTEGER,
    ZKIND INTEGER,
    ZKINDSUBTYPE INTEGER,
    ZLIBRARYSCOPESHARESTATE INTEGER,
    ZORIENTATION INTEGER,
    ZPACKEDACCEPTABLECROPRECT INTEGER,
    ZPACKEDBADGEATTRIBUTES INTEGER,
    ZPACKEDPREFERREDCROPRECT INTEGER,
    ZPLAYBACKSTYLE INTEGER,
    ZPLAYBACKVARIATION INTEGER,
    ZSAVEDASSETTYPE INTEGER,
    ZSEARCHINDEXREBUILDSTATE INTEGER,
    ZSPATIALTYPE INTEGER,
    ZSYNDICATIONSTATE INTEGER,
    ZTHUMBNAILINDEX INTEGER,
    ZTRASHEDREASON INTEGER,
    ZTRASHEDSTATE INTEGER,
    ZVIDEOCPDURATIONVALUE INTEGER,
    ZVIDEOCPVISIBILITYSTATE INTEGER,
    ZVIDEODEFERREDPROCESSINGNEEDED INTEGER,
    ZVIDEOKEYFRAMETIMESCALE INTEGER,
    ZVIDEOKEYFRAMEVALUE INTEGER,
    ZVISIBILITYSTATE INTEGER,
    ZWIDTH INTEGER,
    ZADDITIONALATTRIBUTES INTEGER,
    ZCLOUDFEEDASSETSENTRY INTEGER,
    ZCOMPUTESYNCATTRIBUTES INTEGER,
    ZCOMPUTEDATTRIBUTES INTEGER,
    ZCONVERSATION INTEGER,
    ZDAYGROUPHIGHLIGHTBEINGASSETS INTEGER,
    ZDAYGROUPHIGHLIGHTBEINGEXTENDEDASSETS INTEGER,
    ZDAYGROUPHIGHLIGHTBEINGKEYASSETPRIVATE INTEGER,
    ZDAYGROUPHIGHLIGHTBEINGKEYASSETSHARED INTEGER,
    ZDAYGROUPHIGHLIGHTBEINGSUMMARYASSETS INTEGER,
    ZDAYHIGHLIGHTBEINGCOLLAGEASSETSMIXED INTEGER,
    ZDAYHIGHLIGHTBEINGCOLLAGEASSETSPRIVATE INTEGER,
    ZDAYHIGHLIGHTBEINGCOLLAGEASSETSSHARED INTEGER,
    ZDUPLICATEMETADATAMATCHINGALBUM INTEGER,
    ZDUPLICATEPERCEPTUALMATCHINGALBUM INTEGER,
    ZEXTENDEDATTRIBUTES INTEGER,
    ZHIGHLIGHTBEINGASSETS INTEGER,
    ZHIGHLIGHTBEINGEXTENDEDASSETS INTEGER,
    ZHIGHLIGHTBEINGKEYASSETPRIVATE INTEGER,
    ZHIGHLIGHTBEINGKEYASSETSHARED INTEGER,
    ZHIGHLIGHTBEINGSUMMARYASSETS INTEGER,
    ZIMPORTSESSION INTEGER,
    ZLIBRARYSCOPE INTEGER,
    ZMASTER INTEGER,
    ZMEDIAANALYSISATTRIBUTES INTEGER,
    ZMOMENT INTEGER,
    ZMOMENTSHARE INTEGER,
    ZMONTHHIGHLIGHTBEINGKEYASSETPRIVATE INTEGER,
    ZMONTHHIGHLIGHTBEINGKEYASSETSHARED INTEGER,
    ZPHOTOANALYSISATTRIBUTES INTEGER,
    ZTRASHEDBYPARTICIPANT INTEGER,
    ZYEARHIGHLIGHTBEINGKEYASSETPRIVATE INTEGER,
    ZYEARHIGHLIGHTBEINGKEYASSETSHARED INTEGER,
    Z_FOK_CLOUDFEEDASSETSENTRY INTEGER,
    Z_FOK_DAYHIGHLIGHTBEINGCOLLAGEASSETSPRIVATE INTEGER,
    Z_FOK_DAYHIGHLIGHTBEINGCOLLAGEASSETSMIXED INTEGER,
    Z_FOK_DAYHIGHLIGHTBEINGCOLLAGEASSETSSHARED INTEGER,
    ZADDEDDATE TIMESTAMP,
    ZADJUSTMENTTIMESTAMP TIMESTAMP,
    ZANALYSISSTATEMODIFICATIONDATE TIMESTAMP,
    ZCLOUDBATCHPUBLISHDATE TIMESTAMP,
    ZCLOUDLASTVIEWEDCOMMENTDATE TIMESTAMP,
    ZCLOUDSERVERPUBLISHDATE TIMESTAMP,
    ZCURATIONSCORE FLOAT,
    ZDATECREATED TIMESTAMP,
    ZDURATION FLOAT,
    ZFACEADJUSTMENTVERSION TIMESTAMP,
    ZHDRGAIN FLOAT,
    ZHIGHLIGHTVISIBILITYSCORE FLOAT,
    ZICONICSCORE FLOAT,
    ZLASTSHAREDDATE TIMESTAMP,
    ZLATITUDE FLOAT,
    ZLONGITUDE FLOAT,
    ZMODIFICATIONDATE TIMESTAMP,
    ZOVERALLAESTHETICSCORE FLOAT,
    ZPROMOTIONSCORE FLOAT,
    ZSORTTOKEN FLOAT,
    ZSTICKERCONFIDENCESCORE FLOAT,
    ZTRASHEDDATE TIMESTAMP,
    ZAVALANCHEUUID VARCHAR,
    ZCAPTURESESSIONIDENTIFIER VARCHAR,
    ZCLOUDASSETGUID VARCHAR,
    ZCLOUDBATCHID VARCHAR,
    ZCLOUDCOLLECTIONGUID VARCHAR,
    ZCLOUDOWNERHASHEDPERSONID VARCHAR,
    ZDELETEREASON VARCHAR,
    ZDIRECTORY VARCHAR,
    ZFILENAME VARCHAR,
    ZMEDIAGROUPUUID VARCHAR,
    ZORIGINALCOLORSPACE VARCHAR,
    ZUNIFORMTYPEIDENTIFIER VARCHAR,
    ZUUID VARCHAR,
    ZIMAGEREQUESTHINTS BLOB,
    ZLOCATIONDATA BLOB,
    ZALBUMASSOCIATIVITY INTEGER
)
```
""",
    },
    {
        "question": "What is the table schema for ZCOMPUTEDASSETATTRIBUTES?",
        "answer": """
```sql
CREATE TABLE ZCOMPUTEDASSETATTRIBUTES ( 
    Z_PK INTEGER PRIMARY KEY,
    Z_ENT INTEGER,
    Z_OPT INTEGER,
    ZASSET INTEGER,
    ZBEHAVIORALSCORE FLOAT,
    ZFAILURESCORE FLOAT,
    ZHARMONIOUSCOLORSCORE FLOAT,
    ZIMMERSIVENESSSCORE FLOAT,
    ZINTERACTIONSCORE FLOAT,
    ZINTERESTINGSUBJECTSCORE FLOAT,
    ZINTRUSIVEOBJECTPRESENCESCORE FLOAT,
    ZLIVELYCOLORSCORE FLOAT,
    ZLOWLIGHT FLOAT,
    ZNOISESCORE FLOAT,
    ZPLEASANTCAMERATILTSCORE FLOAT,
    ZPLEASANTCOMPOSITIONSCORE FLOAT,
    ZPLEASANTLIGHTINGSCORE FLOAT,
    ZPLEASANTPATTERNSCORE FLOAT,
    ZPLEASANTPERSPECTIVESCORE FLOAT,
    ZPLEASANTPOSTPROCESSINGSCORE FLOAT,
    ZPLEASANTREFLECTIONSSCORE FLOAT,
    ZPLEASANTSYMMETRYSCORE FLOAT,
    ZSHARPLYFOCUSEDSUBJECTSCORE FLOAT,
    ZTASTEFULLYBLURREDSCORE FLOAT,
    ZWELLCHOSENSUBJECTSCORE FLOAT,
    ZWELLFRAMEDSUBJECTSCORE FLOAT,
    ZWELLTIMEDSHOTSCORE FLOAT
);
```
""",
    },
    {
        "question": "Which photos are from NYC with no face masks?",
        "answer": """
```sql
WITH
NYCPhotos AS (
    SELECT
        Z_PK,
        ZUUID,
        ZDATECREATED,
        ZLATITUDE,
        ZLONGITUDE
    FROM ZASSET
    WHERE
        ZLATITUDE BETWEEN 40.5 AND 40.9
        AND ZLONGITUDE BETWEEN -74.05 AND -73.7
        AND ZTRASHEDSTATE = 0
        AND ZHIDDEN = 0
),
UnmaskedFaces AS (
    SELECT
        ZASSETFORFACE,
        COUNT(*) AS face_count
    FROM ZDETECTEDFACE
    WHERE ZHASFACEMASK = 0
    GROUP BY ZASSETFORFACE
)
SELECT
    n.ZUUID,
    datetime(n.ZDATECREATED + 978307200, 'unixepoch', 'localtime') AS photo_date,
    n.ZLATITUDE,
    n.ZLONGITUDE,
    u.face_count
FROM NYCPhotos n
JOIN UnmaskedFaces u ON n.Z_PK = u.ZASSETFORFACE
ORDER BY n.ZDATECREATED DESC
LIMIT 10;
```
""",
    },
    {
        "question": "Which are the highest aesthetically rated photos of 2024?",
        "answer": """
```sql
SELECT
  ZUUID,
  ZOVERALLAESTHETICSCORE,
  datetime(ZDATECREATED + 978307200, 'unixepoch')
FROM
  ZASSET
WHERE
  ZOVERALLAESTHETICSCORE IS NOT NULL
  AND datetime(ZDATECREATED + 978307200, 'unixepoch') >= '2024-01-01'
  AND datetime(ZDATECREATED + 978307200, 'unixepoch') < '2025-01-01'
  AND ZTRASHEDSTATE = 0
  AND ZHIDDEN = 0
ORDER BY
  ZOVERALLAESTHETICSCORE DESC
LIMIT 10;
```
""",
    },
    {
        "question": "Which photos have the most of my favorite people, all in different places?",
        "answer": """
```sql
WITH PersonFrequency AS (
    -- Get frequency data for all people with names
    SELECT
        p.Z_PK as PersonID,
        p.ZDISPLAYNAME as PersonName,
        p.ZFACECOUNT as FaceCount,
        p.ZPERSONUUID as PersonUUID,
        -- Normalize frequency to 0-1 scale relative to most frequent person
        CAST(p.ZFACECOUNT AS FLOAT) / (SELECT MAX(ZFACECOUNT) FROM ZPERSON) as NormalizedFrequency
    FROM ZPERSON p
    WHERE p.ZFACECOUNT > 0
      AND p.ZDISPLAYNAME IS NOT NULL
      AND p.ZDISPLAYNAME != ''
),
PhotoPeople AS (
    -- Get unique named people in each photo
    SELECT
        a.ZUUID,
        a.Z_PK as AssetID,
        a.ZDATECREATED as DateCreated,
        a.ZLATITUDE as Latitude,
        a.ZLONGITUDE as Longitude,
        df.ZPERSONFORFACE as PersonID
    FROM ZASSET a
    JOIN ZDETECTEDFACE df ON df.ZASSETFORFACE = a.Z_PK
    JOIN ZPERSON p ON p.Z_PK = df.ZPERSONFORFACE
    WHERE df.ZPERSONFORFACE IS NOT NULL
      AND df.ZHIDDEN = 0
      AND df.ZASSETVISIBLE = 1
      AND a.ZTRASHEDSTATE = 0
      AND a.ZHIDDEN = 0
      AND p.ZDISPLAYNAME IS NOT NULL
      AND p.ZDISPLAYNAME != ''
    GROUP BY a.Z_PK, df.ZPERSONFORFACE
),
PhotoScores AS (
    -- Calculate density scores
    SELECT
        pp.ZUUID,
        pp.AssetID,
        pp.DateCreated,
        pp.Latitude,
        pp.Longitude,
        -- Round lat/lng to 0.1 degree
        ROUND(pp.Latitude, 1) as LatRound,
        ROUND(pp.Longitude, 1) as LngRound,
        COUNT(pp.PersonID) as UniquePersonCount,
        SUM(pf.NormalizedFrequency) as TotalFrequencyWeight,
        SUM(pf.NormalizedFrequency) / COUNT(pp.PersonID) as AvgFrequencyPerPerson,
        (SUM(pf.NormalizedFrequency) * COUNT(pp.PersonID)) as DensityScore,
        GROUP_CONCAT(pf.PersonName, ', ') as People
    FROM PhotoPeople pp
    JOIN PersonFrequency pf ON pf.PersonID = pp.PersonID
    GROUP BY pp.ZUUID, pp.AssetID, pp.DateCreated, pp.Latitude, pp.Longitude
    HAVING UniquePersonCount >= 3  -- At least 3 NAMED people in the photo
),
RankedPhotos AS (
    -- Rank photos within each rounded lat/lng group by density score
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY LatRound, LngRound
            ORDER BY DensityScore DESC, DateCreated DESC
        ) as rn
    FROM PhotoScores
)
SELECT
    ZUUID,
    UniquePersonCount,
    TotalFrequencyWeight,
    AvgFrequencyPerPerson,
    DensityScore,
    People,
    Latitude,
    Longitude,
    LatRound,
    LngRound,
    datetime(DateCreated + 978307200, 'unixepoch') as PhotoDate
FROM RankedPhotos
WHERE rn = 1
ORDER BY DensityScore DESC
LIMIT 10;
```
""",
    },
    {
        "question": "Which are the most aesthetic photos of Mom from 2024, all in different places?",
        "answer": """
```sql
WITH PersonID AS (
    SELECT Z_PK
    FROM ZPERSON
    WHERE ZDISPLAYNAME = 'Mom'
),
PhotoAestheticScores AS (
    SELECT
        a.ZUUID,
        a.ZOVERALLAESTHETICSCORE,
        a.ZDATECREATED,
        a.ZLATITUDE,
        a.ZLONGITUDE,
        a.Z_PK as AssetID
    FROM ZASSET a
    JOIN ZDETECTEDFACE df ON df.ZASSETFORFACE = a.Z_PK
    JOIN PersonID p ON p.Z_PK = df.ZPERSONFORFACE
    WHERE
        a.ZTRASHEDSTATE = 0
        AND a.ZHIDDEN = 0
        AND a.ZOVERALLAESTHETICSCORE IS NOT NULL
        AND datetime(a.ZDATECREATED + 978307200, 'unixepoch') >= '2024-01-01'
        AND datetime(a.ZDATECREATED + 978307200, 'unixepoch') < '2025-01-01'
        AND df.ZPERSONFORFACE IS NOT NULL
        AND df.ZHIDDEN = 0
        AND df.ZASSETVISIBLE = 1
),
RankedPhotos AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                ROUND(ZLATITUDE, 1),
                ROUND(ZLONGITUDE, 1)
            ORDER BY ZOVERALLAESTHETICSCORE DESC, ZDATECREATED DESC
        ) as rn
    FROM PhotoAestheticScores
)
SELECT
    ZUUID,
    ZOVERALLAESTHETICSCORE,
    datetime(ZDATECREATED + 978307200, 'unixepoch') as PhotoDate,
    ZLATITUDE,
    ZLONGITUDE
FROM RankedPhotos
WHERE rn = 1
ORDER BY ZOVERALLAESTHETICSCORE DESC
LIMIT 10;
```
""",
    },
    {
        "question": "Which photos are from NYC at night?",
        "answer": """
```sql
WITH NYCPhotos AS (
    SELECT Z_PK, ZUUID, ZDATECREATED, ZLATITUDE, ZLONGITUDE
    FROM ZASSET
    WHERE ZLATITUDE BETWEEN 40.5 AND 40.9
      AND ZLONGITUDE BETWEEN -74.05 AND -73.7
      AND ZTRASHEDSTATE = 0
      AND ZHIDDEN = 0
),
LowLightScores AS (
    SELECT ZASSET, ZLOWLIGHT
    FROM ZCOMPUTEDASSETATTRIBUTES
    WHERE ZLOWLIGHT IS NOT NULL AND ZLOWLIGHT > 0.5
)
SELECT n.ZUUID,
       datetime(n.ZDATECREATED + 978307200, 'unixepoch', 'localtime') AS photo_date,
       n.ZLATITUDE,
       n.ZLONGITUDE,
       l.ZLOWLIGHT
FROM NYCPhotos n
JOIN LowLightScores l ON n.Z_PK = l.ZASSET
ORDER BY l.ZLOWLIGHT DESC, n.ZDATECREATED DESC
LIMIT 10;
```
""",
    }
]


def delete_album_if_exists(album_name):
    """
    Delete an album using AppleScript if it exists.

    Args:
        album_name: Name of the album to delete

    Returns:
        bool: True if album was deleted or didn't exist, False if error occurred
    """
    delete_script = f'''
    tell application "Photos"
        try
            delete album "{album_name}"
            return "deleted"
        on error errMsg
            return "not_found"
        end try
    end tell
    '''

    try:
        result = subprocess.run(['osascript', '-e', delete_script],
                               capture_output=True, text=True, check=False)
        return result.stdout.strip() in ["deleted", "not_found"]
    except Exception:
        return False


def get_system_prompt():
    """
    Returns the system prompt for the AI model with randomly selected examples.
    """

    # Build examples string
    examples = []
    for qa in EXAMPLE_QA_PAIRS:
        examples.append(f"Q: {qa['question']}\nA: {qa['answer']}")

    examples_text = "\n\n".join(examples)

    return f"""
You are a photo search assistant for Apple Photos using the SQLite database to find photos for the user. You are using the Photo database in `~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite`

Answer the user's question with a SQLite query that will fetch their required data, including the `ZUUID` for any photo.

Rules:
* Only answer with SQL, do not include instructions or explanations

These are example responses:
{examples_text}
"""


def text_to_photo(prompt, output_dir=None, album_name=None, api_base="http://localhost:11434/v1", api_key="ollama", model="qwen3-coder-ctx"):
    """
    Convert natural language query to SQL and export matching photos.

    Args:
        prompt: Natural language query about photos
        output_dir: Directory to export photos to (mutually exclusive with album_name)
        album_name: Album name to add photos to (mutually exclusive with output_dir)
        api_base: OpenAI API base URL
        api_key: API key for authentication
        model: Model name to use
    """
    # Initialize OpenAI client
    client = OpenAI(base_url=api_base, api_key=api_key)

    # Initialize Photos database
    photos_path = os.path.expanduser("~/Pictures/Photos Library.photoslibrary")
    photosdb = osxphotos.PhotosDB(photos_path)

    # Get system prompt
    system_prompt = get_system_prompt()

    # Prepare user prompt
    user_prompt = f"Q: {prompt}\nA: "

    # Create messages for API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"Querying AI model: {model}")
    print(f"Prompt: {prompt}\n")

    # Get response from API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2048,
        temperature=0
    )

    output = response.choices[0].message.content
    raw_sql = output.strip().removeprefix("```sql").removesuffix("```").strip()

    print("Generated SQL:")
    print("-" * 50)
    print(raw_sql)
    print("-" * 50)
    print()

    # Execute SQL query
    try:
        df = pd.read_sql_query(raw_sql, photosdb._db_connection)
    except Exception as e:
        print(f"Error executing SQL: {e}")
        return

    if df.empty:
        print("No photos found matching the query.")
        return

    # Get photo UUIDs
    if 'ZUUID' not in df.columns:
        print("Error: Query must return ZUUID column")
        return

    uuids = df.ZUUID
    photos = photosdb.photos_by_uuid(uuids)

    if len(photos) == 0:
        print("No photos found.")
        return

    print(f"\nFound {len(photos)} photos")

    if album_name:
        # Add photos to album
        print(f"Adding to album: {album_name}")
        print("-" * 50)

        # Delete existing album if it exists
        if delete_album_if_exists(album_name):
            print(f"Removed existing album '{album_name}' if it existed")
        else:
            print(f"Warning: Could not delete existing album '{album_name}'")

        # Create new album
        album = PhotosAlbum(album_name)

        # Add all photos to album
        album.extend(photos)
        print(f"Added {len(photos)} photos to album '{album_name}'")

    else:
        # Export to directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Exporting to: {output_path.absolute()}")
        print("-" * 50)

    if not album_name:
        # Export photos
        exported_count = 0
        for i, photo in enumerate(photos, 1):
            if not photo:
                continue

            # Define export options
            options = ExportOptions(
                convert_to_jpeg=True,
                jpeg_quality=0.9,
                download_missing=True,
                overwrite=True
            )

            exporter = PhotoExporter(photo)
            result = exporter.export(output_path, options=options)

            if result.converted_to_jpeg:
                file_path = result.converted_to_jpeg[0]
                exported_count += 1
                print(f"[{i}/{len(photos)}] Exported: {file_path}")
            elif result.exported:
                file_path = result.exported[0]
                exported_count += 1
                print(f"[{i}/{len(photos)}] Exported: {file_path}")
            else:
                print(f"[{i}/{len(photos)}] Failed to export photo: {photo.uuid}")

        print("-" * 50)
        print(
            f"\nExport complete: {exported_count}/{len(photos)} photos exported to {output_path.absolute()}")
    else:
        print("-" * 50)
        print(f"\nAlbum update complete: {len(photos)} photos in album '{album_name}'")

    # Display additional info from dataframe if available
    if len(df.columns) > 1:
        print("\nAdditional query results:")
        print(df.to_string(index=False))


def get_help_examples(num_examples=3):
    """Generate random example commands for help output."""
    # Select random examples for help
    selected_items = random.sample(EXAMPLE_QA_PAIRS, min(num_examples, len(EXAMPLE_QA_PAIRS)))

    examples = []
    for i, qa in enumerate(selected_items):
        if i == 0:
            # First example uses album (default)
            examples.append(f'  %(prog)s -p "{qa["question"]}"')
        elif i == 1:
            # Second example uses album with custom name
            album_name = qa['question'].lower().replace(
                '?', '').replace(' ', '_').replace(',', '')[:20]
            examples.append(f'  %(prog)s -p "{qa["question"]}" -a "{album_name}"')
        else:
            # Third example uses directory export
            dir_name = qa['question'].lower().replace(
                '?', '').replace(' ', '_').replace(',', '')[:30]
            examples.append(f'  %(prog)s -p "{qa["question"]}" -d ./{dir_name}')

    return "\n".join(examples)


def main():
    epilog_text = f"""
Examples:
{get_help_examples(3)}

Environment variables:
  OPENAI_API_BASE: API base URL (default: http://localhost:11434/v1)
  OPENAI_API_KEY: API key (default: ollama)
  OPENAI_MODEL: Model name (default: qwen3-coder:30b)
        """

    parser = argparse.ArgumentParser(
        description="Search Apple Photos using natural language queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text
    )

    parser.add_argument(
        '-p', '--prompt',
        type=str,
        required=True,
        help='Natural language prompt to search for photos'
    )

    # Make -a and -d mutually exclusive
    output_group = parser.add_mutually_exclusive_group()

    output_group.add_argument(
        '-a', '--album',
        type=str,
        nargs='?',
        const='text2photos',
        help='Album name to add photos to (default: text2photos). This is the default if neither -a nor -d is specified.'
    )

    output_group.add_argument(
        '-d', '--directory',
        type=str,
        help='Output directory for exported photos (alternative to album)'
    )

    parser.add_argument(
        '--api-base',
        type=str,
        default=os.getenv('OPENAI_API_BASE', 'http://localhost:11434/v1'),
        help='OpenAI-compatible API base URL'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        default=os.getenv('OPENAI_API_KEY', 'ollama'),
        help='API key for authentication'
    )

    parser.add_argument(
        '--model',
        type=str,
        default=os.getenv('OPENAI_MODEL', 'qwen3-coder:30b'),
        help='Model name to use for query generation'
    )

    args = parser.parse_args()

    # Default to album mode if neither specified
    if not args.directory and args.album is None:
        args.album = 'text2photos'

    try:
        text_to_photo(
            prompt=args.prompt,
            output_dir=args.directory,
            album_name=args.album,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
