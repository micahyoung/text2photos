#!/usr/bin/env python3
"""
Apple Photos search script using OpenAI-compatible API and SQLite queries.
Converts natural language queries to SQL queries for Apple Photos database.
"""

import argparse
import itertools
import os
import random
import subprocess
import pandas as pd
from pathlib import Path
from openai import OpenAI
import osxphotos
from osxphotos import ExportOptions, PhotoExporter, PhotosAlbum

DDLS = [
    """
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
    """
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
    """
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
    """
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
)
```
"""
]

# Example Q&A pairs for the AI model
EXAMPLE_QA_PAIRS = [
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


def get_system_prompt(selected_examples=None, selected_ddls=None):
    """
    Returns the system prompt for the AI model with selected examples and DDLs.

    Args:
        selected_examples: List of example indices to include, or None for all
        selected_ddls: List of DDL indices to include, or None for all
    """
    if selected_examples is None:
        selected_examples = range(len(EXAMPLE_QA_PAIRS))
    if selected_ddls is None:
        selected_ddls = range(len(DDLS))

    # Build examples string
    examples = []
    for idx in selected_examples:
        qa = EXAMPLE_QA_PAIRS[idx]
        examples.append(f"Q: {qa['question']}\nA: {qa['answer']}")

    examples_text = "\n\n".join(examples)

    # Format DDLs with SQL code blocks - only selected ones
    selected_ddls_content = [DDLS[i] for i in selected_ddls]
    ddls_formatted = "\n\n".join(selected_ddls_content)

    return f"""
You are a photo search assistant for Apple Photos using the SQLite database to find photos for the user. You are using the Photo database in `~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite`

Answer the user's question with a SQLite query that will fetch their required data.

Rules:
* Only answer with SQL, do not include instructions or explanations
* Join any tables you need to get the data
* Always include the `ZUUID` column in your SELECT statement

Table DDLS:
{ddls_formatted}

Example responses:
{examples_text}
"""


def text_to_photo(prompt, output_dir=None, album_name=None, api_base="http://localhost:11434/v1", api_key="ollama", model="qwen3-coder-ctx", num_examples=-1, num_ddls=-1):
    """
    Convert natural language query to SQL and export matching photos.

    Args:
        prompt: Natural language query about photos
        output_dir: Directory to export photos to (mutually exclusive with album_name)
        album_name: Album name to add photos to (mutually exclusive with output_dir)
        api_base: OpenAI API base URL
        api_key: API key for authentication
        model: Model name to use
        num_examples: Number of examples to use in combinations (-1 means all)
        num_ddls: Number of DDLs to use in combinations (-1 means all)
    """

    # Initialize OpenAI client
    client = OpenAI(base_url=api_base, api_key=api_key, max_retries=0)

    # Initialize Photos database
    photos_path = os.path.expanduser("~/Pictures/Photos Library.photoslibrary")
    photosdb = osxphotos.PhotosDB(photos_path)

    # Determine batch sizes
    if num_examples == -1:
        example_batch_size = len(EXAMPLE_QA_PAIRS)
    else:
        example_batch_size = min(num_examples, len(EXAMPLE_QA_PAIRS))

    if num_ddls == -1:
        ddl_batch_size = len(DDLS)
    else:
        ddl_batch_size = min(num_ddls, len(DDLS))

    # Generate all combinations of both DDLs and examples
    example_indices = list(range(len(EXAMPLE_QA_PAIRS)))
    ddl_indices = list(range(len(DDLS)))

    example_combinations = list(
        itertools.combinations(example_indices, example_batch_size))
    ddl_combinations = list(itertools.combinations(ddl_indices, ddl_batch_size))

    # Create all combinations of DDL and example pairs
    combinations = list(itertools.product(
        ddl_combinations, example_combinations))

    print(f"Querying AI model: {model}")
    print(f"Prompt: {prompt}")
    print(
        f"Trying {len(combinations)} combination(s) of {ddl_batch_size} DDL(s) and {example_batch_size} example(s)\n")

    # Try each combination until we find a valid query with results
    valid_result = None
    valid_sql = None
    valid_df = None

    for i, (ddl_combo, example_combo) in enumerate(combinations, 1):
        print(
            f"Attempt {i}/{len(combinations)}: Using DDLs {list(ddl_combo)} and examples {list(example_combo)}")

        # Get system prompt with selected DDLs and examples
        system_prompt = get_system_prompt(example_combo, ddl_combo)

        # Prepare user prompt
        user_prompt = f"Q: {prompt}\nA: "

        # Create messages for API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Get response from API
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0
            )

            output = response.choices[0].message.content
            raw_sql = output.strip().removeprefix("```sql").removesuffix("```").strip()

            # Try to execute the SQL query
            df = pd.read_sql_query(raw_sql, photosdb._db_connection)

            # Check if we have results and ZUUID column
            if not df.empty and 'ZUUID' in df.columns:
                print(f"  ✓ Valid SQL with {len(df)} results found")
                valid_result = True
                valid_sql = raw_sql
                valid_df = df
                break
            elif df.empty:
                print(f"  ✗ Query returned no results")
            else:
                print(f"  ✗ Query missing ZUUID column")

        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            continue

    if not valid_result:
        print("\nNo valid query found after trying all combinations.")
        return

    # Use the valid query result
    print("\nUsing successful query:")
    print("-" * 50)
    print(valid_sql)
    print("-" * 50)
    print()

    df = valid_df

    # Get photo UUIDs
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
        print(
            f"\nAlbum update complete: {len(photos)} photos in album '{album_name}'")

    # Display additional info from dataframe if available
    if len(df.columns) > 1:
        print("\nAdditional query results:")
        print(df.to_string(index=False))


def get_help_examples(num_examples=3):
    """Generate random example commands for help output."""
    # Select random examples for help
    selected_items = random.sample(
        EXAMPLE_QA_PAIRS, min(num_examples, len(EXAMPLE_QA_PAIRS)))

    examples = []
    for i, qa in enumerate(selected_items):
        if i == 0:
            # First example uses album (default)
            examples.append(f'  %(prog)s -p "{qa["question"]}"')
        elif i == 1:
            # Second example uses album with custom name
            album_name = qa['question'].lower().replace(
                '?', '').replace(' ', '_').replace(',', '')[:20]
            examples.append(
                f'  %(prog)s -p "{qa["question"]}" -a "{album_name}"')
        else:
            # Third example uses directory export
            dir_name = qa['question'].lower().replace(
                '?', '').replace(' ', '_').replace(',', '')[:30]
            examples.append(
                f'  %(prog)s -p "{qa["question"]}" -o ./{dir_name}')

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

    # Make -a and -o mutually exclusive
    output_group = parser.add_mutually_exclusive_group()

    output_group.add_argument(
        '-a', '--album',
        type=str,
        nargs='?',
        const='text2photos',
        help='Album name to add photos to (default: text2photos). This is the default if neither -a nor -o is specified.'
    )

    output_group.add_argument(
        '-o', '--output-dir',
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

    parser.add_argument(
        '-e', '--num-examples',
        type=int,
        default=-1,
        help='Number of examples to use in combinations (-1=all examples). Will try all combinations until finding valid results.'
    )

    parser.add_argument(
        '-d', '--num-ddls',
        type=int,
        default=-1,
        help='Number of DDLs to use in combinations (-1=all DDLs). Will try all combinations until finding valid results.'
    )

    args = parser.parse_args()

    # Default to album mode if neither specified
    if not args.output_dir and args.album is None:
        args.album = 'text2photos'

    try:
        text_to_photo(
            prompt=args.prompt,
            output_dir=args.output_dir,
            album_name=args.album,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            num_examples=args.num_examples,
            num_ddls=args.num_ddls
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
